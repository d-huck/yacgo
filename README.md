# yacgo

This is the codebase for the YacGo project create for CS394R Reinforcement Learning at UT Austin Spring 2024.

## Requirements and Setup

This project requires [ZeroMQ](https://zeromq.org) for passing messages between GPU workers and gameplay workers. For MacOs, it is installable using homebrew:

```
brew install zeromq
```

For Debian based systems you can install it using apt:

```
apt-get install libzmq3-dev
```

For other installations, refer to the [ZeroMQ github repository](https://github.com/zeromq/libzmq).

To install all requirements for this project, first set up your favorite python environment management and install:

```
pip install torch
python setup.py install
```

## Running the code

This project breaks training into two processes. The first process handles policy and value inference, training, evaluation, and replay buffer management. The second process controls gameplay workers which choose moves using MCTS search. ZeroMQ is utilized to pass messages between the workers. To run the training process

```
python yacgo/scripts/stack.py --training_steps_per_epoch 1000 --num_servers 2 --board_size 3
```

In another terminal window, run the gameplay worker:

```
python yacgo/scripts/client.py --pcap_train 400 --pcap_fast 100 --num_servers 2 --board_size 3
```
