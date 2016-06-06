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
cmd:option('-hidden', 200, 'hidden size')
cmd:option('-episodes', 10, 'number of episodes to simulate')
cmd:option('-quiet', false, 'produce less output')
cmd:option('-momentum', 0.5, 'momentum')
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
stay = 2
numActions = 3
gameSize = params['size']
emulator.parameterize(gameSize, speeds, params['max-time'])

-- Return a tensor with the first row corresponding to the current
-- scene, and the subsequent rows corresponding to x(t+1) - x(t) difference
-- frames.
function phi(history)
  local x = torch.Tensor(histLen, history[1].size)
  -- Copy the last `histLen` scenes into x
  x[1]:copy(history[#history].scene)
  for i=2,histLen do
    local prev = #history - (i - 1)
    if prev < 1 then
      prev = 1
    end
    x[i]:copy(history[prev].scene - history[prev+1].scene)
  end
  return x
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

function train(par)
  local learningRate = par['rate']
  local momentum = par['momentum']
  local M = par['episodes']
  local T = par['max-time']+5
  local gamma = par['gamma']
  local weightDecay = par['L2-regularization']
  local beta = par['entropy-regularization']
  local updateEvery = par['update-every']
  local model = makeNet(par)
  -- Only regularize non-bias parameters
  local wdMask = regularizationMask(model.net)
  local wd = wdMask:mul(weightDecay)
  local theta = model.net.par
  local gradTheta = model.net.gradPar
  local runningAdvantage = 0
  local upCount = 0
  local gradCount = 0
  local policyGrad = torch.Tensor(numActions)
  local valueGrad = torch.Tensor(1)
  local averagePolicy = torch.zeros(3)
  local entropyGrad = torch.Tensor(3)
  local r
  gradTheta:zero()
  for i=1,M do
    -- Run the game forward enough to get histLen images
    local s = {emulator.firstEnvironment()}
    for t=2,histLen do
      s[t], r = emulator.evolve(s[t-1], stay)
      s[t-1].r = r
      s[t].xt = phi(s)
      s[t].policy, s[t].V = unpack(model.net:forward(s[t].xt))
    end
    local t=histLen
    while true do
      if t > T or s[t].gameOver then
        print("final reward: " .. s[t-1].r)
        break
      end
      local tStart = t
      while true do
        if s[t].xt == nil then
          s[t].xt = phi(s)
        end
        s[t].policy, s[t].V = unpack(model.net:forward(s[t].xt))
        -- Sample from the policy using the cumulative distribution.
        local policy = s[t].policy:exp()
        averagePolicy:mul(0.99):add(policy)
        print("policy")
        print(policy:view(1,-1))
        print(averagePolicy:view(1,-1))
        s[t].action = samplePolicy(policy)
  
        -- Perform the action.
        s[t+1], r = emulator.evolve(s[t], s[t].action)
        s[t].r = r
  
        print("> " .. s[t].picture .. " took " .. s[t].action .. " to " .. s[t+1].picture ..
          ', reward: ' .. s[t].r .. " episode " .. i .. " timestep " .. t)
  
        if (s[t+1].gameOver or t+1-tStart == updateEvery) then
          -- Compute discounted rewards and accuulate gradients.
          local R
          if s[t+1].gameOver then
            R = 0
          else
            s[t+1].xt = phi(s)
            s[t+1].policy, s[t+1].V = unpack(model.net:forward(s[t+1].xt))
            R = s[t+1].V[1]
          end
          local advantage
          for k=t,tStart,-1 do
            R = s[k].r + gamma * R
            local logPolicy, V = unpack(model.net:forward(s[k].xt))
            print("policy:")
            local policy = torch.exp(logPolicy)
            print(policy:view(1,-1))
            print("entropy: " .. -policy:cmul(logPolicy):sum())
            policyGrad:zero()
            advantage = R - s[k].V[1]
            policyGrad[s[k].action] = advantage
            print("policyGrad only:")
            print(policyGrad:view(1,-1))
            valueGrad[1] = advantage
            print("value error: " .. advantage^2)
            entropyGrad:exp(logPolicy)
            entropyGrad:cmul(torch.add(logPolicy, 1))
            print("entropyGrad:")
            print(entropyGrad:view(1,-1))
            policyGrad:add(-beta, entropyGrad)
            model.net:backward(s[k].xt, {policyGrad,valueGrad})
          end
          gradCount = gradCount + 1

          if gradCount % 100 == 0 then
            gradTheta:div(100)
            -- Update the parameters
            -- Apply weight decay
            gradTheta:add(torch.cmul(wd, theta))
            local update = torch.zeros(theta:size(1))
            -- Use momentum, but scaling down the update vector if it's to big, this
            -- helps with exploding gradients.
            -- We're doing a maximization problem, so we add a positive gradient.
            update:mul(momentum):add(learningRate, gradTheta)
            local norm = update:norm()
            if norm > 1 then
              update:mul(1/norm)
            end
            theta:add(update)
            runningAdvantage = 0.99 * runningAdvantage + 0.01 * advantage
            upCount = upCount + 1
            print("[" .. i .. "," .. upCount .. "] runningAdvantage: " .. runningAdvantage ..
              ", advantage: " .. advantage ..  ", update: " .. update:norm() ..
              ", R: " .. R .. ", |theta|: " .. theta:norm())
            gradTheta:zero()
          end

          t = t + 1
          tStart = t
          break
        end
        t = t + 1
      end
    end
--    if i % 500 == 1 then
--      local fn = params.prefix .. "-" .. i .. ".t7"
--      torch.save(fn, model)
--    end
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
      s[t] = emulator.evolve(s[t-1], stay)
    end
    for t=histLen,T do
      s[t].xt = phi(s)
      s[t].policy, s[t].V = unpack(model.net:forward(s[t].xt))
      local policy = s[t].policy:exp()
      s[t].action = samplePolicy(policy)

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


