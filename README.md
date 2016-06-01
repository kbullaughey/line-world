Training with these parameters results in near perfect play:

    ./toy.lua -mode train -size 7 -episodes 4000 > train.out

Testing can be done as follows:

    ./toy.lua -mode test -size 7 -save model.t7 -episodes 1000 -quiet
