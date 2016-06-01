Training with these parameters results in perfect play:

    ./toy.lua -initial-epsilon 0.2 -decay-epsilon-over 2000 -rate 0.01 -simple -mode train -batch 64 -size 7 -prefix try2 -hidden 200 -episodes 5000 -gamma 0.97 > try2.out

Testing can be done as follows:

    ./toy.lua -simple -mode test -size 7 -save try2.t7 -hidden 200 -episodes 1000 -quiet
