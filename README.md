# EulerStateNetworks
This repository contains the TensorFlow 2.0 / Keras implementation of Euler State Networks (EuSN), as described in the paper
C. Gallicchio, "Euler State Networks: Non-dissipative Reservoir Computing", Submitted to Journal (2023) [https://arxiv.org/abs/2203.09382] 

## Files
Currently, two files are provided
* `euler.py`, which contains the main classes definition, including the EulerReservoirCell, the EuSN, and all the recurrent layers and architectures (both Reservoir Computing-based and fully trainable) used in the experiments
* `experiments_RC.py`, which contains the code for running the experiments on the benchmarks with all the reservoir methods (EuSN, ESN, R-ESN) used in the paper;
* `experiments_RC_MNIST.py`, which contains the code for running the experiments on the benchmarks with all the reservoir methods (EuSN, ESN, R-ESN) used in the paper for the sequential MNIST task (the only difference is the usage of a buffering approach while computing the reservoir states, to keep the one-shot training of the readout)
* `experiments_trainable.py`, which contains all the code for running the experiments on the benchmarks with all the fully trainable models (GRU, A-RNN, RNN) used in the paper

## Datasets

The pool of datasets used in the paper can be downloaded from the following link 
https://www.dropbox.com/scl/fo/0kbqezo00vye79sk3lvs5/h?rlkey=vqu0hkglqphj00s2nf4n692kv&dl=0
