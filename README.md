# Overview

This repo contains a Reinforcement Learning demo that was the basis for my own learning about RL. I built a very small grid-world game that can be played interactively or used as an emulator for producing training data for RL. I also provide neural net implementations that can easily master this game.

I started by learning about Q-learning followed the [Learning to play Atari][1] paper and taught myself RL from [Sutton's book][2]. I then tried out n-step Q-learning and policy-network methods based on [Asynchronous Methods for Deep Reinforcement Learning][3] and with help from [Andrej Karpathy's post on RL][4].

[1]: http://arxiv.org/pdf/1312.5602v1.pdf
[2]: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node1.html
[3]: http://arxiv.org/pdf/1602.01783v1.pdf
[4]: http://karpathy.github.io/2016/05/31/rl/

The machine learning part is written in Lua using Torch, and the game is made playable in a web browser using a javascript server running on node that talks to the Lua backend over websockets.

# Game design

The game consists of a linear world with two docks on either side of a river. There is a boat that goes back and forth across the river at constant velocity, reflecting off the docks. The player starts on one dock and must wait for the boat to reach that dock, hop on the boat, and wait until the boat reaches the other side before hopping off onto the far dock.

The game state is parameterized by the following:

0. Size of the world in grid tiles (e.g. 10).
0. Direction and speed of the boat (which can vary from episode to episode).
0. Position the player.
0. Position of the boat.

There are two representations of this game, an alphabetic visualization for humans and a numeric representation for the computer. In the former, each tile is given a letter, in the latter each tile is given a number:

    w: water (unoccupied)  -3
    b: boat  (unoccupied)  -2
    d: dock  (unoccupied)  -1
    D: dock  (occupied)     1
    B: boat  (occupied)     2
    W: water (occupied)     3

Here's an example starting state:

    dwwwwwbwwD

The corresponding numerical vector would be:

    (-1, -3, -3, -3, -3, -3, -2, -3, -3, 1)

You can think of this like an image with one row of pixels.

This means that the player is on the right-hand dock (hence the capital letter). If the player were to jump into the water you would see:

    dwwwwwbwWd

And this would be the end of the epoch.

There are three possible terminal reward outcomes and one intermediate reward:

0. -2: Falling in the water
0. -1: Running out of time
0. +10: Getting to the far dock
0. +2: Getting on the boat (intermediate reward)

At any given time step, there are three possible actions: left, right, and stay.

Here's an example gameplay:

    Dwwwwwwbwd
    Dwwwwwbwwd
    Dwwwwwbwwd
    Dwwwwbwwwd
    Dwwwwbwwwd
    Dwwwbwwwwd
    Dwwwbwwwwd
    Dwwbwwwwwd
    Dwwbwwwwwd
    Dwbwwwwwwd
    Dwbwwwwwwd
    Dbwwwwwwwd
    dBwwwwwwwd
    Dbwwwwwwwd
    dBwwwwwwwd
    dwBwwwwwwd
    dwBwwwwwwd
    dwwBwwwwwd
    dwwBwwwwwd
    dwwwBwwwwd
    dwwwBwwwwd
    dwwwwBwwwd
    dwwwwBwwwd
    dwwwwwBwwd
    dwwwwwBwwd
    dwwwwwwBwd
    dwwwwwwBwd
    dwwwwwwwBd
    dwwwwwwwbD

# Playing the game

You can play the game interactively in a web browser.

You'll need to start the lua/torch backend process:o

    ./q-learning.lua -mode play

Then, concurrently you'll need to start the node server (this assumes you have run `npm install` in the `client` sub-directory):

    cd client
    node index.js

The lua backend listens on port 2601 for the browser, and the browser serves the interactive game on port 2600. Then you can go to [localhost:2600](http://localhost:2600/) to play the game.

# Modeling details

This RL toy uses Q-learning, and thus the neural net serves as a function approximator to `Q(s,a)` where `s` is the state and `a` is the action. In practice, the neural net outputs three values, one for each possible action. Thus only one forward pass is needed to compute the `Q` values for the three possible actions.

The neural net gets only a partial representation of the full state. Specifically, it gets a tensor of size (K,L) where K is the number of image frames to present and L is the length of the line world. Each row corresponds to a 1D image as described above. The first row is the actual image, and the subsequent rows are image differences (i.e., `x[t+t] - x[t]`).

Importantly, it doesn't know the speed of the boat or anything about what the goal of the game is or anything about rivers, docks, boats, drowning, or the like.

The neural net is a simple feed-forward network with one hidden layer and Tanh non-linearities. It treats the whole image as one long vector and is fully connected.

# Training a model

## One-step Q-learning

Training with these parameters results in decent play (~95% win rate):

    ./q-learning.lua -mode train -episodes 6000 > train.out

Testing can be done as follows:

    ./q-learning.lua -mode test -episodes 1000 -quiet

One of the more challenging aspects of this game is that the speed varies. If a single speed is used, we get a 100% win rate (1000/1000):

    ./q-learning.lua -mode train -episodes 2500 -save one-speed.t7 -speeds 0.5 > one-speed.out
    ./q-learning.lua -mode test -save one-speed.t7 -episodes 1000 -speeds 0.5 -quiet

One challenge of one-step Q-learning is that until we have a pretty good
estimate of Q(s,a), we really only learn something from the time steps that
result in rewards. We can thus speed up training sampling time steps for the
minibatch in a non-uniform way. Specifically, we can use rejection sampling to
reject the bulk of the non-reward steps.

    ./q-learning.lua -mode train -episodes 3000 -reject 0.8 > train.out

Rejection sampling works particularly well in this game, because at least in
the case of getting on the boat, falling in the water, and getting off at the
far dock, the reward is connected directly to actions taken in that step. 

## n-step Q-learning

I have also implemented n-step q-learning as described in
[Asynchronous methods for deep reinforcement learning](https://arxiv.org/abs/1602.01783), Algorithm 2, although my method is not asynchronous.

    ./n-step-q-learning.lua -mode train -episodes 6000 -n 2 \
        -rate 0.02 > n-step-train.out

And then testing it as follows:

    ./n-step-q-learning.lua -mode test -episodes 1000 -quiet

I got 947 of 1000 wins. And if we stick to a single speed, we get 100% win rate.

## Vanilla policy rollouts

On-policy methods, such as policy gradient methods, offer an alternative to the off-policy methods of Q-learning. The policy-network variety work by parameterizing the policy with a neural network such that the output of the neural network is a probability distribution over actions and then sampling from this discrete distribution to play the game. After the game terminates, discounted rewards are computed for each step, and using backpropagation one can compute parameter updates, thereby allowing one to train the policy network.

Unlike the temporal different algorithms, we cannot use a replay memory because older rollouts were played using a different policy, and thus are not samples from the policy under consideration and using them would mess up our estimate of the gradient.

So instead we can roll out several policies and average the gradients. This can be parallelized and thus I use the term agent, but in this implementation the concurrent agents are not truly parallel (as lua is generally limited to one thread).

Without the replay memory, and using full trajectories, we need a larger number of episodes to get this to train:

    ./policy-rollout.lua -mode train -rate 0.1 -episodes 50000 -entropy-regularization 0.3 \
        -norm 4 -agents 100 -speeds 0.4,0.6 > policy-rollout.out

One modification that seems to help is including a regularization term that is proportional to the entropy of the policy. I have also simplified the problem a bit by specifying only one speed for the boat. 

    ./policy-rollout.lua -mode test -episodes 1000 -speeds 0.4,0.6 -quiet

And we get 1000/1000 wins.

## Actor-critic

In the actor critic on-policy method, we use a policy network like the vanilla policy rollout method above, but with two differences:

1. In addition to our network emitting a policy probability distribution, we also emit an estimate of the value of the current state: `V(s)`. We use this value as a baseline, or critic, against which to compute an advantage `A = R - V(s)`. Optimizing this has lower variance that directly optimizing the return.
2. We don't perform rollouts all the way to the end of an episode. Instead we perform a more limited rollout (e.g., 6) and use our value estimate as a proxy for the return at a non-terminal step.

We can train with less data and using the default three speeds (0.3,0.5,0.6):

    ./actor-critic.lua -mode train -rate 0.1 -episodes 30000 -entropy-regularization 0.2 \
        -norm 4 -agents 100 > actor-critic.out

And we can test the resulting trained model:

    ./actor-critic.lua -mode test -episodes 1000 -quiet

And we get 948/1000 wins.

# Watching the AI play

To watch a trained model play, launch the lua backend using the `simulate` mode and specify your saved model:

    ./q-learning.lua -mode simulate -save model.t7

Then after making sure the node server is running, open the web browser
to [localhost:2600/ai](http://localhost:2600/ai).


