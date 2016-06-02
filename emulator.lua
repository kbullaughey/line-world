local emu = {}

emu.sceneLetters = {[-3]="w", [-2]="b", [-1]="d", "D", "B", "W"}
emu.winReward = 10

-- Pretty print a scene using letters
function emu.sceneString(scene)
  local s = ""
  for i=1,scene:size(1) do
    s = s .. emu.sceneLetters[scene[i]]
  end
  return s
end

function emu.waterTiles(env)
  return env.size - 2
end

-- Discrete position of the boat.
function emu.boatTile(env)
  return 2+math.floor(env.boat*emu.waterTiles(env))
end

-- Return a tensor representing the visual display.
function emu.renderScene(env)
  local scene = torch.Tensor(env.size):fill(-3)
  -- Add the docks
  scene[1] = -1
  scene[env.size] = -1
  -- Put the boat into the scene.
  scene[emu.boatTile(env)] = -2
  -- Put the player into the scene.
  scene[env.player] = math.abs(scene[env.player])
  return scene
end

-- Set some invariant emulator properties
function emu.parameterize(gameSize, speeds, maxTime)
  emu.gameSize = gameSize
  emu.speeds = speeds
  emu.maxTime = maxTime
end

-- Initial state of the game
function emu.firstEnvironment()
  -- Randomly select whether we start on 1 and end on 10 or the opposite
  local start, finish =
    unpack(torch.Tensor{1,emu.gameSize}:index(1,torch.randperm(2):long()):totable())
  local env = {
    -- Width of the scene in tiles
    size = emu.gameSize,
    -- The player's position in [1,size]
    player = start,
    goal = finish,
    -- Don't draw the boat or player yet
    scene = torch.Tensor{1,-3,-3,-3,-3,-3,-3,-3,-3,-1},
    -- speed is how fast the boat moves in tiles/tick.
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
  env.speed = emu.speeds[torch.randperm(emu.speeds:size(1))[1]]
  -- Switch direction if prob 0.5
  if torch.uniform() < 0.5 then
    env.direction = -1
  end
  -- Clock ticks
  env.ticks = 0
  -- Render the scene
  env.scene = emu.renderScene(env)
  -- Also privide the string representation
  env.picture = emu.sceneString(env.scene)
  return env
end

function emu.reward(env, oldEnv)
  -- Determine the reward based on where the player is.
  local p = emu.sceneLetters[env.scene[env.player]]
  local oldP = emu.sceneLetters[oldEnv.scene[oldEnv.player]]

  local r = 0
  if env.player == env.goal then
    r = emu.winReward
  elseif p == "W" then
    r = -2
  elseif env.ticks > emu.maxTime then
    r = -1
  elseif p == "B" and oldP ~= "B" then
    r = 1
  end
  return r
end

-- Advance the game forward one click.
-- action can be left,stay,right coded as 1,2,3
-- returns a new env, and a reward.
-- Reward is -2 (fall in water), -1 (time expires), 0, or 1 (win)
-- If the reward is negative, the game is over.
function emu.evolve(env, action)
  local z = {size=env.size, speed=env.speed, goal=env.goal}
  -- Calculate the players new position.
  z.player = env.player + (action-2)
  if z.player <= 0 then
    z.player = 1
  elseif z.player > z.size then
    z.player = z.size
  end
  -- See if the player is on the boat.
  local onBoat = emu.sceneLetters[env.scene[z.player]] == "B"
  -- Calculate the new position of the boat.
  z.boat = env.boat + env.speed/emu.waterTiles(env)*env.direction
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
  -- If the player is on the boat, update his position to the boat's new position.
  if onBoat then
    z.player = emu.boatTile(z)
  end
  -- Render the new scene.
  z.scene = emu.renderScene(z)
  z.picture = emu.sceneString(z.scene)
  -- Update the clock
  z.ticks = env.ticks + 1
  local r = emu.reward(z, env)
  if r < 0 or r == emu.winReward then
    z.gameOver = true
  end
  return z, r
end

return emu
