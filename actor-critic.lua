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
cmd:option('-mode', '', 'one of train|test')
cmd:option('-size', 10, 'length of world')
cmd:option('-agents', 50, 'number of concurrent agents')
cmd:option('-hidden', 200, 'hidden size')
cmd:option('-episodes', 10, 'number of episodes to simulate')
cmd:option('-quiet', false, 'produce less output')
cmd:option('-momentum', 0.5, 'momentum')
cmd:option('-norm', 1, 'maximum update norm')
cmd:option('-max-time', 75, 'maximum length of games in clock ticks.')
cmd:option('-rate', 0.01, 'learning rate')
cmd:option('-save', '', 'Filename to save model to (or read from when -mode test).')
cmd:option('-gamma', 0.90, 'discounting parameter, gamma')
cmd:option('-L2-regularization', 1e-05, 'weight-decay regularization')
cmd:option('-entropy-regularization', 0.01, 'weight for policy entropy regularization')
cmd:option('-prefix', 'model', 'saved model prefix')
cmd:option('-speeds', '0.3,0.5,0.7', 'different speeds of the boat')
cmd:option('-frames', 6, 'number of frames to include')
cmd:option('-update-every', 6, 'how often to perform updates')
params = cmd:parse(arg)

startedAt = os.time()
speeds = tablex.map(function(x) return tonumber(x) end, stringx.split(params.speeds, ","))
speeds = torch.Tensor(speeds)
histLen = params['frames']
numActions = 3
gameSize = params['size']
emulator.parameterize(gameSize, speeds, params['max-time'], histLen)

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
  ns.policy = nn.LogSoftMax()(nn.Linear(par.hidden, numActions)(ns.h))
  ns.value = nn.Linear(par.hidden, 1)(ns.h)
  ns.net = nn.gModule({ns.x}, {ns.policy,ns.value})
  ns.net.par, ns.net.gradPar = ns.net:getParameters()
  ns.net.par:uniform(-0.05, 0.05)
  ns.net.gradPar:zero()
  return ns
end

-- Sample from the policy using the cumulative distribution.
function samplePolicy(policy)
  local action = 1
  local cdf = policy:cumsum()
  local u = torch.uniform()
  while cdf[action] < u do
    action = action + 1
  end
  return action
end

-- Create game images enumerating all possible positions of the boat and player.
-- For each of these, create a game image and feed it through the neural net.
function examinePolicyByEnumeratingStates(model, size, speed, goal, direction)
  -- Tile centers on uniform (0,1) interval
  local waterTiles = emulator.waterTiles({size=size})
  local waterTileCenters = torch.range(0.5, waterTiles-0.5, 1.0):div(waterTiles)
  local softmax = nn.SoftMax()
  local playerPositions = {1, "boat", size}
  for b=1,waterTiles do
    local boatRel = waterTileCenters[b]
    for p=1,#playerPositions do
      local player
      if playerPositions[p] == "boat" then
        player = emulator.boatTile({size=size, boat=boatRel})
      else
        player = playerPositions[p]
      end
      local env = emulator.customEnvironment(player, goal, speed, direction, boatRel)
      local xt = emulator.phiBackward(env, histLen)
      print("> " .. env.picture)
      print(xt)
      local policy, V = unpack(model.net:forward(xt))
      print(softmax:forward(policy):view(1,-1))
      print("V: " .. V[1])
    end
  end
end

function weightDecayNormedUpdate(par, grad, wd, rate)
  -- Apply weight decay
  grad:add(torch.cmul(wd, par))
  local norm = grad:norm()
  if norm > params['norm'] then
    grad:mul(params['norm']/norm)
  end
  print("grad:norm(): " .. grad:norm())
  par:add(rate, grad)
end

function train(par)
  local learningRate = par['rate']
  local momentum = par['momentum']
  local M = par['episodes']
  local T = par['max-time']+5
  local gamma = par['gamma']
  local weightDecay = par['L2-regularization']
  local beta = par['entropy-regularization']
  local updateEvery = par['update-every']
  local numAgents = par['agents']

  -- All agents will share this model. Because lua isn't truely parallelized.
  -- All concurrent coroutines run in one thread.
  local model = makeNet(par)
  local theta = model.net.par
  local numPar = theta:size(1)
  local sharedTheta = torch.Tensor(numPar):copy(theta)
  local gradTheta = model.net.gradPar
  local valueGradTheta = torch.zeros(numPar)
  local policyGradTheta = torch.zeros(numPar)
  local gradCount = 0
  local completed = 0

  -- Create a number of agents, each which will maintain an environment and
  -- a set of parameters, and pass back a gradient to be averaged.
  local agents = {}
  for g=1,numAgents do
    agents[g] = coroutine.create(function(identity)
      -- get set up and then yield.
      local r
      local policyGrad = torch.Tensor(numActions)
      local valueGrad = torch.Tensor(1)
      local entropyGrad = torch.Tensor(3)
      print("agent " .. identity .. " started")
      coroutine.yield()
      print("agent " .. identity .. " resumed (1)")
      while true do
        -- Run the game forward enough to get histLen images
        print("agent " .. identity .. " starting episode")
        local s = {emulator.firstEnvironment()}
        for t=2,histLen do
          s[t], r = emulator.evolve(s[t-1], emulator.stay)
          s[t-1].r = r
          s[t].xt = emulator.phi(s)
          s[t].logPolicy, s[t].V = unpack(model.net:forward(s[t].xt))
        end
        local t=histLen
        while true do
          if s[t].gameOver then
            print("final reward: " .. s[t-1].r)
            completed = completed + 1
            break
          end
--          local tStart = t
          while true do
            if s[t].xt == nil then
              s[t].xt = emulator.phi(s)
            end
            s[t].logPolicy, s[t].V = unpack(model.net:forward(s[t].xt))
            -- Sample from the policy using the cumulative distribution.
            local policy = torch.exp(s[t].logPolicy)
            s[t].action = samplePolicy(policy)
      
            -- Perform the action.
            s[t+1], r = emulator.evolve(s[t], s[t].action)
            s[t].r = r
      
            print("> " .. s[t].picture .. " took " .. s[t].action .. " to " .. s[t+1].picture ..
              ', reward: ' .. s[t].r .. " timestep " .. t)
            print(policy:view(1,-1))
      
            t = t + 1
--            if (s[t].gameOver or t-tStart == updateEvery) then
            if (s[t].gameOver) then
              -- Compute discounted rewards and accuulate gradients.
              local R = 0
--              if s[t].gameOver then
--                print("agent " .. identity .. " game over at " .. t)
--                R = 0
--              else
--                s[t].xt = emulator.phi(s)
--                s[t].policy, s[t].V = unpack(model.net:forward(s[t].xt))
--                R = s[t].V[1]
--              end
              local advantage
              for k=t-1,histLen,-1 do
                R = s[k].r + gamma * R
                advantage = R
                gradTheta:zero()
                local logPolicy, V = unpack(model.net:forward(s[k].xt))
                --advantage = R - s[k].V[1]
                print("R: " .. R .. ", V: " .. s[k].V[1], " advantage: " .. advantage)
                -- Do one packward pass for the policy with the valueGrad zeroed out.
                -- Accumulate the gradient in policyGradTheta
                policyGrad:zero()
                policyGrad[s[k].action] = advantage
                valueGrad[1] = 0
                entropyGrad:exp(logPolicy)
                entropyGrad:cmul(torch.add(logPolicy, 1))
                policyGrad:add(-beta, entropyGrad)
                model.net:backward(s[k].xt, {policyGrad, valueGrad})
                policyGradTheta:add(gradTheta)
                -- And one backward pass for the value with the policyGrad zeroed out.
                -- Accumulate the gradient in valueGradTheta
--                gradTheta:zero()
--                valueGrad[1] = -2 * advantage
--                valueGrad[1] = -2 * advantage
--                policyGrad:zero()
--                model.net:backward(s[k].xt, {policyGrad,valueGrad})
--                valueGradTheta:add(gradTheta)
              end
              gradCount = gradCount + 1
              break
            end
          end
          coroutine.yield()
          print("agent " .. identity .. " resumed (2)")
          print(s[t-1])
          if s[t-1].logPolicy then
            print("% " .. s[t-1].action .. " " ..
              stringx.join(" ", torch.exp(s[t-1].logPolicy):totable()))
            print(torch.exp(model.net:forward(s[t-1].xt)[1]):view(1,-1))
            print(torch.exp(model.net:forward(emulator.phi(s))[1]):view(1,-1))
          end
        end
      end
    end)
    -- Call the agent once to get set up and get to the first yield.
    coroutine.resume(agents[g], g)
  end

  -- Only regularize non-bias parameters
  local wdMask = regularizationMask(model.net)
  local wd = wdMask:mul(weightDecay)

  -- Look until we've done at least M episodes. Actual number will be slightly
  -- higher so that it is a multiple of numAgents.
  print("All agents started")
  local upCount = 0
  local picks = 0
  while completed < M do
    local whichAgent = picks % numAgents + 1
    local ok, err = coroutine.resume(agents[whichAgent], whichAgent)
    if not ok then
      print(err)
      error("agent " .. whichAgent .. " failed to resume")
    end
    if whichAgent == numAgents then
      -- After a full pass through the agents we update our gradient
      policyGradTheta:div(numAgents)
      --valueGradTheta:div(numAgents)

      -- Update the parameters
      weightDecayNormedUpdate(theta, policyGradTheta, wd, learningRate)
      --weightDecayNormedUpdate(theta, valueGradTheta, wd, -learningRate)
      upCount = upCount + 1
      print("[" .. completed .. "," .. upCount .. "] |theta|: " .. theta:norm())
      gradTheta:zero()
      policyGradTheta:zero()
      valueGradTheta:zero()
    end
    picks = picks + 1
  end
  return model
end

function test(model, par)
  local M = par['episodes']
  local T = par['max-time']+5
  local gamesWon = 0
  for i=1,M do
    local s = {emulator.firstEnvironment()}
    for t=2,histLen do
      s[t] = emulator.evolve(s[t-1], emulator.stay)
    end
    for t=histLen,T do
      s[t].xt = emulator.phi(s)
      s[t].logPolicy, s[t].V = unpack(model.net:forward(s[t].xt))
      local _, best = s[t].logPolicy:max(1)
      s[t].action = best[1]

      -- Perform the action.
      s[t+1], s[t].r = emulator.evolve(s[t], s[t].action)

      if not par['quiet'] then
        print("> " .. s[t].picture .. " took " .. s[t].action .. " to " .. s[t+1].picture ..
          ', reward: ' .. s[t].r)
      end
      -- If the game is over, start the next episode
      if s[t+1].gameOver then
        if s[t].r == emulator.winReward then
          gamesWon = gamesWon + 1
        end
        if not par['quiet'] then
          print("final reward: " .. s[t].r .. " iter " .. t)
        end
        break
      end
    end
  end
  print("Won " .. gamesWon .. " out of " .. M)
end

if params.mode == "train" then
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
else
  print("nothing to do.")
end

endedAt = os.time()
print("execution took " .. endedAt - startedAt .. " seconds")


