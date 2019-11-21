# Extended Markov Games (EMGs)
Experimental settings to test EMGs as an extended markov decision process to train multi-agent reinforcement learning algorithms to solve multi-agent specifications

## Installation instructions

You might clone this repository by running:
	git clone https://github.com/bgLeon/EMG.git

This repository requires [Python3.5](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow](https://www.tensorflow.org/), and [sympy](http://www.sympy.org). 

## Running examples

To run any of the algorithms, execute *run_experiments.py*. This code receives 3 parameters: The RL algorithm to use (which might be "i-dqn-l" or "i-lpopl"), the tasks to solve (which might be "sequence", "interleaving", "safety"), the maximum number of steps for the agent or agents to learn, an option to choose running multi-agent or single-agent (a boleen True or False for multi-agent), and a render option to render the map with the agents while learning. For instance, the following command solves the 10 *sequence specifications* using I-LPOPL with multiple agents:

    python3 run_experiments.py --algorithm="i-lpopl" --tasks="sequence" 

To run the same configuration for the single agent experiment just add --singleA

The results will be printed and saved in './tmp'

## Acknowledgments

Our implementations and experiments are based and extended from the code provided by [Icarte et al.](https://bitbucket.org/RToroIcarte/lpopl.git).