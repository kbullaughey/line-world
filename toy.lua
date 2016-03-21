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
cmd:option('-max-time', 200, 'maximum length of games in clock ticks.')
params = cmd:parse(arg)

sceneLetters = {[-3]="w", [-2]="b", [-1]="d", "D", "B", "W"}
speeds = {0.2,0.25,0.5}
histLen = 4
stay = 2
epsilon = 1
numActions = 3
maxTime = params['max-time']
gameSize = params['size']

-- Pretty print a scene using letters
function sceneString(scene)
  local s = ""
  for i=1,scene:size(1) do
    s = s .. sceneLetters[scene[i]]
  end
  return s
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
  local env = {
    -- Width of the scene in tiles
    size = gameSize,
    -- The player's position in [1,size]
    player = 1,
    -- Don't draw the boat or player yet
    scene = torch.Tensor{1,-3,-3,-3,-3,-3,-3,-3,-3,-1},
    -- speed is how fast the boat moves in tiles/tick. Can be 0.1, 0.2, or 0.3.
    speed = 0,
    -- Boat direction can be either 1 (right) or -1 (left)
    direction = 1,
    -- Actual boat position in the interval (0,1). This excludes the docks.
    boat = 0,
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

function reward(env)
  -- Determine the reward based on where the player is.
  local p = sceneLetters[env.scene[env.player]]
  local r = 0
  if env.player == env.size then
    r = 1
  elseif p == "W" then
    r = -2
  elseif env.ticks > maxTime then
    r = -1
  end
  return r
end

-- Advance the game forward one click.
-- action can be left,stay,right coded as 1,2,3
-- returns a new env, and a reward.
-- Reward is -2 (fall in water), -1 (time expires), 0, or 1 (win)
-- If the reward is negative, the game is over.
function evolve(env, action)
  local z = {size=env.size, speed=env.speed}
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
  return z, reward(z)
end

-- Interactive episode
function play()
  local copas = require 'copas'
  local ws = require 'websocket'
  local config = {
    port = 2601,
    protocols = {
      lineworld = function(ws)
        local game = firstEnvironment()
        local r
        local command
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
          end
          game, r = evolve(game, command)
          ws:send("> " .. game.picture)
          command = 2
          if r ~= 0 then
            break
          end
        end
        local rewardMessage
        if r == -2 then
          rewardMessage = "You drowned."
        elseif r == -1 then
          rewardMessage = "Time up."
        elseif r == 1 then
          rewardMessage = "You won!"
        else
          error("Invalid reward")
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
  local x = torch.Tensor(history[1].size, histLen)
  if #history < histLen then
    error("Insufficient history")
  end
  -- Copy the last `histLen` scenes into x
  for i=#history-histLen+1, #history do
    x:select(2,i):copy(history[i].scene)
  end
  return x
end

-- The seqLenghs will be a batchSize-length vector of the constant, histSize. This
-- is an artifact of how I've implemented GRUChain.
seqLengths = torch.Tensor(params.batch):fill(histLen)

-- Sample minibatch with replacement
function minibatch(D, batchSize)
  -- This will be three tensors, x(t), x(t+1), actions, rewards, seqLengths
  local example = {
    torch.Tensor(batchSize, histLen, gameSize),
    torch.Tensor(batchSize, histLen, gameSize),
    torch.Tensor(batchSize),
    torch.Tensor(batchSize),
    seqLengths,
  }
  for b=1,batchSize do
    local w = math.floor(torch.uniform() * #D) + 1
    local xt, a, r, xtp1 = unpack(D[w])
    example[1][b]:copy(xt)
    example[2][b]:copy(xtp1)
    example[3][b] = a
    example[4][b] = a
  end
  return example
end

function makeNet(par)
  local ns = {}
  -- The net should receive a tuple {x,seqLengths}
  ns.inTuple = nn.Identity()
  ns.chainMod = lstm.GRUChain(gameSize, {par.hidden}, histLen)
  ns.chainOut = ns.chainMod(ns.inTuple)
  ns.Q = nn.Linear(par.hidden, numActions)
  ns.net = nn.gModule({ns.inTuple}, {ns.Q})
  -- Need to reenable sharing after getParameters(), which broke my sharing.
  ns.net.par, ns.net.gradPar = ns.net:getParameters()
  ns.chainMod:setupSharing()
  ns.net.par:uniform(-0.05, 0.05)
  ns.net.gradPar:zero()
  return ns
end

-- This will return a vector of scores over actions.
function Q(x, net)
  return net:forward(x)
end

function train(par)
  local D = {}
  local M = par['episodes']
  local T = par['max-time']+5
  local capacity = par['capacity']
  local batchSize = par['batch']
  -- Current index into D
  local d = 1
  local model = makeNet(par)
  for i=1,M do
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
      if u < epsilon then
        -- Random action
        action = math.floor(torch.uniform(1,4))
      else
        local q = Q(xt, model)
        local _, bestQ = q:max(1)
      end
      s[t+1], r = evolve(s[t], action)
      xtp1 = phi(s)
      D[d] = {xt, action, r, xtp1}
      d = (d % capacity) + 1
      local example = minibatch(D, batchSize)
    end
  end
end

if params.mode == 'play' then
  play()
elseif params.mode == "train" then
  train(params)
else
  print("nothing to do.")
end

-- END
