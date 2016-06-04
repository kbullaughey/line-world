#!/usr/bin/env th

-- This game takes place in a linear world described by a vector of length 10.
-- It represents a river with a dock on each side and a boat somewhere in the
-- river. The goal is to cross the river on the boat. The boat moves back and
-- forth at some rate unknown to the player, but observable over time.
--
-- Each component of the vector corresponds to one of the following possible
-- states:
--    w: water (unoccupied) -3
--    b: boat (unoccupied) -2
--    d: dock (unoccupied) -1
--    D: dock (occupied) 1
--    B: boat (occupied) 2
--    W: water (occupied) 3
--
-- Capital letters are used when the player is at that position. Thus there are
-- six possible values at each position. The numbers are used by the game, while
-- the letters are used for rending a more interpretable representation of the
-- game state.
--
-- An environment has the following structure:
--
-- (size, speed, player, boat, direction, goal)

require 'nngraph'
emulator = require './emulator'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Toy game for reinfocement learning')
cmd:text()
cmd:text('Options')
cmd:option('-mode', '', 'one of play|train|test|simulate')
cmd:option('-batch', 64, 'batch size')
cmd:option('-size', 10, 'length of world')
cmd:option('-hidden', 200, 'hidden size')
cmd:option('-capacity', 5000, 'sample memory capacity')
cmd:option('-episodes', 10, 'number of episodes to simulate')
cmd:option('-quiet', false, 'produce less output')
cmd:option('-momentum', 0.5, 'momentum')
cmd:option('-max-time', 75, 'maximum length of games in clock ticks.')
cmd:option('-rate', 0.01, 'learning rate')
cmd:option('-save', '', 'Filename to save model to (or read from when -mode test).')
cmd:option('-initial-epsilon', 0.1, 'initial random choice probability, epsilon')
cmd:option('-final-epsilon', 0.02, 'final random choice probability, epsilon')
cmd:option('-decay-epsilon-over', 1000, 'Number of episodes over which to decay epsilon')
cmd:option('-gamma', 0.97, 'discounting parameter, gamma')
cmd:option('-regularization', 1e-05, 'weight-decay regularization')
cmd:option('-prefix', 'model', 'saved model prefix')
cmd:option('-speeds', '0.3,0.5,0.7', 'different speeds of the boat')
cmd:option('-frames', 6, 'number of frames to include')
cmd:option('-update-every', 1, 'How often to update the parameters (int >= 1)')
params = cmd:parse(arg)

startedAt = os.time()
speeds = tablex.map(function(x) return tonumber(x) end, stringx.split(params.speeds, ","))
speeds = torch.Tensor(speeds)
histLen = params['frames']
stay = 2
initialEpsilon = params['initial-epsilon']
finalEpsilon = params['final-epsilon']
decayOver = params['decay-epsilon-over']
numActions = 3
gameSize = params['size']
emulator.parameterize(gameSize, speeds, params['max-time'])

-- Interactive episode
function play(simulator)
  local copas = require 'copas'
  local ws = require 'websocket'
  local config = {
    port = 2601,
    protocols = {
      lineworld = function(ws)
        local game = emulator.firstEnvironment()
        print(game)
        local r
        local command
        local history = {}
        print("have client")
        while true do
          local message = ws:receive()
          if message == "left" then
            print("left")
            command = 1
          elseif message == "right" then
            print("right")
            command = 3
          elseif message == "stay" then
            command = 2
          elseif message == "propose" then
            if #history < histLen then
              command = 2
            else
              local x = phi(history)
              local q = Q(x, simulator.net)
              local _, bestQ = q:max(2)
              command = bestQ[1][1]
            end
          end
          game, r = emulator.evolve(game, command)
          table.insert(history, game)
          ws:send("> " .. game.picture)
          if command ~= 2 then
            print(game)
          end
          command = 2
          if game.gameOver then
            break
          end
        end
        local rewardMessage
        if r == -2 then
          rewardMessage = "You drowned."
        elseif r == -1 then
          rewardMessage = "Time up."
        elseif r == emulator.winReward then
          rewardMessage = "You won!"
        end
        ws:send(rewardMessage)
        ws:close()
      end
    }
  }
  local server = ws.server.copas.listen(config)
  copas.loop()
end

-- Return a tensor with the first row corresponding to the current
-- scene, and the subsequent rows corresponding to x(t+1) - x(t) difference
-- frames.
function phi(history)
  local x = torch.Tensor(histLen, history[1].size)
  if #history < histLen then
    error("Insufficient history")
  end
  -- Copy the last `histLen` scenes into x
  x[1]:copy(history[#history].scene)
  for i=2,histLen do
    local prev = #history - (i - 1)
    x[i]:copy(history[prev].scene - history[prev+1].scene)
  end
  return x
end

-- Sample minibatch with replacement
function minibatch(D, batchSize)
  -- This will be three tensors, x(t), x(t+1), actions, rewards
  local example = {
    torch.Tensor(batchSize, histLen, gameSize),
    torch.Tensor(batchSize, histLen, gameSize),
    -- This will be a one-hot encoding.
    torch.zeros(batchSize,numActions),
    torch.Tensor(batchSize),
  }
  for b=1,batchSize do
    local w, xt, a, r, xtp1
    local accept = false
    while not accept do
      w = math.floor(torch.uniform() * #D) + 1
      xt, a, r, xtp1 = unpack(D[w])
      if r == 10 then
        accept = true
      else
        if torch.uniform() < 0.1 then
          accept = true
        end
      end
    end
    example[1][b]:copy(xt)
    example[2][b]:copy(xtp1)
    example[3][b][a] = 1
    example[4][b] = r
  end
  return example
end

-- Only regularize linear transform weight parameters, not bias parameters. I make
-- this judgement based on whether the parameter tensor is 1D or 2D.
function regularizationMask(net)
  local mask = torch.zeros(net.par:size(1))
  local params = net:parameters()
  local offset = 1
  for i=1,#params do
    local dims = params[i]:dim()
    if dims == 2 then
      local mx_size = params[i]:size(1) * params[i]:size(2)
      mask:narrow(1, offset, mx_size):fill(1)
      offset = offset + mx_size
    else
      local len = params[i]:size(1)
      offset = offset + len
    end
  end
  if offset-1 ~= net.par:size(1) then
    error("unexpected length")
  end
  return mask
end

function makeNet(par)
  local ns = {}
  ns.x = nn.Identity()()
  ns.xr = nn.Reshape(gameSize*histLen)(ns.x)
  ns.h = nn.Tanh()(nn.Linear(gameSize*histLen, par.hidden)(ns.xr))
  ns.Q = nn.Linear(par.hidden, numActions)(ns.h)
  ns.net = nn.gModule({ns.x}, {ns.Q})
  ns.net.par, ns.net.gradPar = ns.net:getParameters()
  ns.net.par:uniform(-0.05, 0.05)
  ns.net.gradPar:zero()
  -- Additional step to condition on action
  ns.givenActionInput = nn.Identity()()
  ns.givenActionMasked = nn.CMulTable()(ns.givenActionInput)
  ns.givenActionSum = nn.Sum(1, 1)(ns.givenActionMasked)
  ns.givenAction = nn.gModule({ns.givenActionInput},{ns.givenActionSum})
  return ns
end

-- This will return a vector of scores over actions.
function Q(x, net)
  if x:dim() == 2 then
    x = x:view(1,histLen,gameSize)
  end
  return net:forward(x)
end

function test(model, par)
  local M = par['episodes']
  local T = par['max-time']+5
  local gamesWon = 0
  for i=1,M do
    local s = {emulator.firstEnvironment()}
    for t=2,histLen do
      s[t] = emulator.evolve(s[t-1], stay)
    end
    for t=histLen,T do
      local xt = phi(s)
      local q = Q(xt, model.net)
      local _, bestQ = q:max(2)
      local action = bestQ[1][1]
      s[t+1], r = emulator.evolve(s[t], action)
      if not par['quiet'] then
        print("> " .. s[t].picture .. " took " .. action .. " to " .. s[t+1].picture ..
          ', reward: ' .. r)
        print(nn.SoftMax():forward(q))
      end
      -- If the game is over, start the next episode
      if s[t+1].gameOver then
        if r == emulator.winReward then
          gamesWon = gamesWon + 1
        end
        if not par['quiet'] then
          print("final reward: " .. r .. " iter " .. t)
        end
        break
      end
    end
  end
  print("Won " .. gamesWon .. " out of " .. M)
end

function train(par)
  local learningRate = par['rate']
  local momentum = par['momentum']
  local D = {}
  local M = par['episodes']
  local T = par['max-time']+5
  local updateFreq = par['update-every']
  local capacity = par['capacity']
  local batchSize = par['batch']
  -- Current index into D
  local d = 1
  local model = makeNet(par)
  local theta = model.net.par
  local gradTheta = model.net.gradPar
  -- Only regularize non-bias parameters
  local weightDecay = par['regularization']
  local gamma = par['gamma']
  local wdMask = regularizationMask(model.net)
  local wd = wdMask:mul(weightDecay)
  local criterion = nn.MSECriterion()
  local runningError = 0
  -- Just for debugging traces
  local epsilon = initialEpsilon
  for i=1,M do
    if i < decayOver then
      epsilon = initialEpsilon + (i-1)*(finalEpsilon - initialEpsilon)/decayOver
    end
    -- Run the game forward enough to get histLen images
    local s = {emulator.firstEnvironment()}
    for t=2,histLen do
      s[t] = emulator.evolve(s[t-1], stay)
    end
    local xt
    local xtp1
    local r
    for t=histLen,T do
      xt = phi(s)
      local u = torch.uniform()
      local action
      local wasRandom = ''
      local q = nil
      if u < epsilon then
        -- Random action
        action = math.floor(torch.uniform(1,4))
        wasRandom = 'R'
      else
        q = Q(xt, model.net)
        local _, bestQ = q:max(2)
        action = bestQ[1][1]
      end
      s[t+1], r = emulator.evolve(s[t], action)
      print("> " .. s[t].picture .. " took " .. action .. wasRandom .. " to " ..
        s[t+1].picture .. ', reward: ' .. r .. " episode " .. i .. " timestep " .. t)
      xtp1 = phi(s)
      D[d] = {xt, action, r, xtp1}
      d = (d % capacity) + 1
      if (d-1) % updateFreq == 0 then
        local example = minibatch(D, batchSize)
        -- Since the targets are a function of the parameters, we compute those first
        -- before the predictions so we don't mess up our nerual net state during
        -- the forward and backward passes.
        local targets = Q(example[2], model.net):max(2)
        local nonTerminal =
          torch.gt(torch.eq(example[4],0):double() + torch.eq(example[4],1):double(),0):double()
        local discountedFuture = nonTerminal:cmul(targets):mul(gamma)
        local y = example[4] + discountedFuture
        gradTheta:zero()
        -- Forward pass
        local predictions = model.net:forward(example[1])
        local givenAction = model.givenAction:forward({predictions, example[3]})
        local err = criterion:forward(givenAction, y)
        -- Backward pass
        local g = criterion:backward(givenAction, y)
        g = model.givenAction:backward({predictions, example[3]}, g)
        model.net:backward(example[1], g[1])
  
        -- Apply weight decay
        gradTheta:add(torch.cmul(wd, theta))
        local update = torch.zeros(theta:size(1))
        -- Use momentum, but scaling down the update vector if it's to big, this
        -- helps with exploding gradients.
        update:mul(momentum):add(-learningRate, gradTheta)
        local norm = update:norm()
        if norm > 1 then
          update:mul(1/norm)
        end
        theta:add(update)
        runningError = 0.99 * runningError + 0.01 * err
        print("runningError: " .. runningError .. ", err: " .. err ..
          ", update: " .. update:norm() .. ", |theta|: " .. theta:norm())
      end

      -- If the game is over, start the next episode
      if s[t+1].gameOver then
        print("final reward:" .. r)
        break
      end
    end
    if i % 500 == 1 then
      local fn = params.prefix .. "-" .. i .. ".t7"
      torch.save(fn, model)
    end
  end
  return model
end

if params.mode == 'play' then
  play()
elseif params.mode == "train" then
  model = train(params)
  if params.save == '' then
    params.save = params.prefix .. ".t7"
  end
  torch.save(params.save, model)
elseif params.mode == "test" then
  if params.save == '' then
    error("Must speficy -save <t7 file>")
  end
  model = torch.load(params.save)
  test(model, params)
elseif params.mode == "simulate" then
  if params.save == '' then
    error("Must speficy -save <t7 file>")
  end
  model = torch.load(params.save)
  play(model)
else
  print("nothing to do.")
end

endedAt = os.time()
print("execution took " .. endedAt - startedAt .. " seconds")

-- END
