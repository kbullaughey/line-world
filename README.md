# Overview

This repo contains a Reinforcement Learning demo that was the basis for my own learning about Q-learning based RL. I built a very small grid-world game that can be played interactively or used as an emulator for producing training data for RL. I also provide a neural net implementation that can easily master this game.

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

This means that the player is on the right-hand dock (hence the capital letter). If the player were to jump into the water you would see:

    dwwwwwbwWd

And this would be the end of the epoch.

There are three possible terminal reward outcomes and one intermediate reward:

0. -2: Falling in the water
0. -1: Running out of time
0. +10: Getting to the far dock
0. +1: Getting on the boat (intermediate reward)

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

    ./toy.lua -mode play

Then, concurrently you'll need to start the node server (this assumes you have run `npm install` in the `client` sub-directory):

    cd client
    node index.js

The lua backend listens on port 2601 for the browser, and the browser serves the interactive game on port 2600. Then you can go to [localhost:2600](http://localhost:2600/) to play the game.

# Training a model

Training with these parameters results in decent play (~95% win rate):

    ./toy.lua -mode train -episodes 4000 > train.out

Testing can be done as follows:

    ./toy.lua -mode test -save model.t7 -episodes 1000 -quiet

# Watching the AI play

To watch a trained model play, launch the lua backend using the `simulate` mode and specify your saved model:

    ./toy.lua -mode simulate -save model.t7

Then after making sure the node server is running, open the web browser
to [localhost:2600](http://localhost:2600/ai).


