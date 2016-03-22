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
-- (size, speed, player, boat, direction,

require 'lstm'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Toy game for reinfocement learning')
cmd:text()
cmd:text('Options')
cmd:option('-mode', 'play|train', 'start a server for playing in a web browser or training')
cmd:option('-batch', 8, 'batch size')
cmd:option('-size', 10, 'length of world')
cmd:option('-hidden', 20, 'GRU hidden size')
cmd:option('-capacity', 5000, 'sample memory capacity')
cmd:option('-episodes', 10, 'number of episodes to simulate')
cmd:option('-momentum', 0.5, 'momentum')
cmd:option('-max-time', 75, 'maximum length of games in clock ticks.')
cmd:option('-rate', 0.1, 'learning rate')
cmd:option('-save', 'model.t7', 'Filename to save model to.')
cmd:option('-initial-epsilon', 0.1, 'initial random choice probability, epsilon')
cmd:option('-final-epsilon', 0.02, 'final random choice probability, epsilon')
cmd:option('-decay-epsilon-over', 500, 'Number of episodes over which to decay epsilon')
cmd:option('-gamma', 0.95, 'discounting parameter, gamma')
cmd:option('-regularization', 1e-05, 'weight-decay regularization')
cmd:option('-prefix', 'model', 'saved model prefix')
params = cmd:parse(arg)

sceneLetters = {[-3]="w", [-2]="b", [-1]="d", "D", "B", "W"}
--speeds = {0.2,0.25,0.5}
speeds = {0.5,0.5}
histLen = 4
stay = 2
winReward = 10
initialEpsilon = params['initial-epsilon']
finalEpsilon = params['final-epsilon']
decayOver = params['decay-epsilon-over']
numActions = 3
maxTime = params['max-time']
gameSize = params['size']
gamma = params['gamma']
actionLabels = {"left", "stay", "right"}

-- Pretty print a scene using letters
function sceneString(scene)
  local s = ""
  for i=1,scene:size(1) do
    s = s .. sceneLetters[scene[i]]
  end
  return s
end

function dumpVec(label, v)
--  print(label .. ":")
--  if v:size(1) > 20 then
--    v = v:narrow(1,1,20)
--  end
--  print(v:view(1,-1))
end

function waterTiles(env)
  return env.size - 2
end

-- Discrete position of the boat.
function boatTile(env)
  return 2+math.floor(env.boat*waterTiles(env))
end

-- Return a tensor representing the visual display.
function renderScene(env)
  local scene = torch.Tensor(env.size):fill(-3)
  -- Add the docks
  scene[1] = -1
  scene[env.size] = -1
  -- Put the boat into the scene.
  scene[boatTile(env)] = -2
  -- Put the player into the scene.
  scene[env.player] = math.abs(scene[env.player])
  return scene
end

-- Initial state of the game
function firstEnvironment()
  -- Randomly select whether we start on 1 and end on 10 or the opposite
  local start, finish =
    unpack(torch.Tensor{1,gameSize}:index(1,torch.randperm(2):long()):totable())
  local env = {
    -- Width of the scene in tiles
    size = gameSize,
    -- The player's position in [1,size]
    player = start,
    goal = finish,
    -- Don't draw the boat or player yet
    scene = torch.Tensor{1,-3,-3,-3,-3,-3,-3,-3,-3,-1},
    -- speed is how fast the boat moves in tiles/tick. Can be 0.1, 0.2, or 0.3.
    speed = 0,
    -- Boat direction can be either 1 (right) or -1 (left)
    direction = 1,
    -- Actual boat position in the interval (0,1). This excludes the docks.
    boat = 0,
    gameOver = false,
  }
  -- Pick the boats position.
  env.boat = torch.uniform()
  -- Pick a speed
  env.speed = speeds[math.floor(torch.uniform(1,3))]
  -- Switch direction if prob 0.5
  if torch.uniform() < 0.5 then
    env.direction = -1
  end
  -- Clock ticks
  env.ticks = 0
  -- Render the scene
  env.scene = renderScene(env)
  -- Also privide the string representation
  env.picture = sceneString(env.scene)
  return env
end

function reward(env, oldEnv)
  -- Determine the reward based on where the player is.
  local p = sceneLetters[env.scene[env.player]]
  local oldP = sceneLetters[oldEnv.scene[oldEnv.player]]

  local r = 0
  if env.player == env.goal then
    r = winReward
  elseif p == "W" then
    r = -2
  elseif env.ticks > maxTime then
    r = -1
  elseif p == "B" and oldEnv ~= "B" then
    r = 1
  end
  return r
end

-- Advance the game forward one click.
-- action can be left,stay,right coded as 1,2,3
-- returns a new env, and a reward.
-- Reward is -2 (fall in water), -1 (time expires), 0, or 1 (win)
-- If the reward is negative, the game is over.
function evolve(env, action)
  local z = {size=env.size, speed=env.speed, goal=env.goal}
  -- See if the player is on the boat.
  local onBoat = sceneLetters[env.scene[env.player]] == "B"
  -- Calculate the new position of the boat.
  z.boat = env.boat + env.speed/waterTiles(env)*env.direction
  -- Reflect the boat away from the edges if necessary, in which case
  -- we switch the direction.
  if z.boat < 0 then
    z.boat = (-1) * z.boat
    z.direction = (-1) * env.direction
  elseif z.boat > 1 then
    z.boat = 2 - z.boat
    z.direction = (-1) * env.direction
  else
    z.direction = env.direction
  end
  -- If the player was on the boat, update his position to the boat's new position.
  if onBoat then
    z.player = boatTile(z)
  else
    z.player = env.player
  end
  -- Calculate the players new position.
  z.player = z.player + (action-2)
  if z.player <= 0 then
    z.player = 1
  elseif z.player > z.size then
    z.player = z.size
  end
  -- Render the new scene.
  z.scene = renderScene(z)
  z.picture = sceneString(z.scene)
  -- Update the clock
  z.ticks = env.ticks + 1
  local r = reward(z, env)
  if r < 0 or r == winReward then
    z.gameOver = true
  end
  return z, r
end

-- Interactive episode
function play(simulator)
  local copas = require 'copas'
  local ws = require 'websocket'
  local config = {
    port = 2601,
    protocols = {
      lineworld = function(ws)
        local game = firstEnvironment()
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
          game, r = evolve(game, command)
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
        elseif r == winReward then
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

-- Update x, shifting one vector off the left side, and adding the new vector
-- on the right.
function phi(history)
  local x = torch.Tensor(histLen, history[1].size)
  if #history < histLen then
    error("Insufficient history")
  end
  -- Copy the last `histLen` scenes into x
  for i=1,histLen do
    x[i]:copy(history[#history-histLen+i].scene)
  end
  return x
end

-- The seqLenghs will be a batchSize-length vector of the constant, histSize. This
-- is an artifact of how I've implemented GRUChain.
seqLengths = torch.Tensor(params.batch):fill(histLen)

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
  -- The net should receive a tuple {x,seqLengths}
  ns.inTuple = nn.Identity()()
  ns.chainMod = lstm.GRUChain(gameSize, {par.hidden}, histLen)
  ns.chainOut = ns.chainMod(ns.inTuple)
  ns.Q = nn.Linear(par.hidden, numActions)(ns.chainOut)
  ns.net = nn.gModule({ns.inTuple}, {ns.Q})
  -- Need to reenable sharing after getParameters(), which broke my sharing.
  ns.net.par, ns.net.gradPar = ns.net:getParameters()
  ns.chainMod:setupSharing()
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
  local lengths = seqLengths:narrow(1,1,x:size(1))
  return net:forward({x,lengths})
end

function train(par)
  local learningRate = par['rate']
  local momentum = par['momentum']
  local D = {}
  local M = par['episodes']
  local T = par['max-time']+5
  local capacity = par['capacity']
  local batchSize = par['batch']
  -- Current index into D
  local d = 1
  local model = makeNet(par)
  local theta = model.net.par
  local gradTheta = model.net.gradPar
  -- Only regularize non-bias parameters
  local weightDecay = par['regularization']
  local wdMask = regularizationMask(model.net)
  local wd = wdMask:mul(weightDecay)
  local criterion = nn.MSECriterion()
  local runningError = 0
  -- Just for debugging traces
  local softmax = nn.SoftMax()
  local epsilon = initialEpsilon
  for i=1,M do
    if i < decayOver then
      epsilon = initialEpsilon + (i-1)*(finalEpsilon - initialEpsilon)/decayOver
    end
    print("episode: " .. i .. ", epsilon: " .. epsilon)
    -- Run the game forward enough to get histLen images
    local s = {firstEnvironment()}
    for t=2,histLen do
      s[t] = evolve(s[t-1], stay)
    end
    local xt
    local xtp1
    local r
    for t=histLen,T do
      xt = phi(s)
      local u = torch.uniform()
      local action
      local wasRandom = ''
      if u < epsilon then
        -- Random action
        action = math.floor(torch.uniform(1,4))
        wasRandom = 'R'
      else
        local q = Q(xt, model.net)
        dumpVec("softmax(q)", softmax:forward(q))
        local _, bestQ = q:max(2)
        action = bestQ[1][1]
      end
      print("> " .. s[t].picture .. " : " .. action .. wasRandom)
      s[t+1], r = evolve(s[t], action)
      print("> " .. s[t+1].picture .. ', reward: ' .. r)
      xtp1 = phi(s)
      D[d] = {xt, action, r, xtp1}
      d = (d % capacity) + 1
      local example = minibatch(D, batchSize)
      -- Since the targets are a function of the parameters, we compute those first
      -- before the predictions so we don't mess up our nerual net state during
      -- the forward and backward passes.
      local targets = Q(example[2], model.net):max(2)
      dumpVec("reward", example[4])
      local nonTerminal =
        torch.gt(torch.eq(example[4],0):double() + torch.eq(example[4],1):double(),0):double()
      dumpVec("nonTerminal", nonTerminal)
      print("num non-terminal: " .. nonTerminal:sum())
      local discountedFuture = nonTerminal:cmul(targets):mul(gamma)
      dumpVec("discountedFuture", discountedFuture)
      local y = example[4] + discountedFuture
      dumpVec("y", y)
      gradTheta:zero()
      -- Forward pass
      local predictions = model.net:forward({example[1],seqLengths})
      local givenAction = model.givenAction:forward({predictions, example[3]})
      dumpVec("givenAction", givenAction)
      if torch.eq(example[4], 1):double():sum() > 0 then
        print("batch includes getting on boat")
        print(predictions:narrow(1,1,8))
      end
      local err = criterion:forward(givenAction, y)
      -- Backward pass
      local g = criterion:backward(givenAction, y)
      dumpVec("gradGivenAction", g)
      g = model.givenAction:backward({predictions, example[3]}, g)
      model.net:backward({example[1],seqLengths}, g[1])

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
  torch.save(params.save, model)
elseif params.mode == "simulate" then
  model = torch.load(params.save)
  play(model)
else
  print("nothing to do.")
end

-- END