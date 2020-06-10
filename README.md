# CSC480Gym

## dqn.py
Instruction to setup and run the deep-q-network agent for LunarLander.

### Setup instruction:
Create/choose an environment using Python >= 3.7.
Ensure the following dependencies are present in your environment:
* gym
* tensorflow
* keras
* box2d-py

[Optional] Note the location of the picke for a pre-trained agent

### Run instructions:
Run the program using the following command:

`dqn.py [-h] [-n NUMBER] [-f FILEPATH] {train,eval}`

e.g.
`python3 dqn.py eval -n 10 -f dqn.pickle`

To run the program in evaluation mode using an agent saved in a file in the current directory named `dqn.pickle` for `10` iterations.

Positional Arguments:
* {train, eval}  Specify whether to run the agent in training or evaluation mode

Optional Arguments:
* -n NUMBER  Number of iterations to run the agent (Default: 100)
* -f FILEPATH The path to save/load the agent pickle (Default: None)
* -h  Show help message and exit
