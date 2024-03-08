# EulerStateNetworks
This repository contains the TensorFlow 2.0 / Keras implementation of Euler State Networks (EuSN), as described in the paper
Gallicchio, Claudio. "Euler state networks: Non-dissipative reservoir computing." Neurocomputing (2024): 127411.
https://www.sciencedirect.com/science/article/pii/S0925231224001826

## Files
Currently, two files are provided
* `euler.py`, which contains the main classes definition, including the EulerReservoirCell, the EuSN, and all the recurrent layers and architectures (both Reservoir Computing-based and fully trainable) used in the experiments
* `experiments_RC.py`, which contains the code for running the experiments on the time-series classification benchmarks with all the reservoir methods (EuSN, ESN, R-ESN, DeepESN) used in the paper;
* `experiments_RC_MNIST.py`, which contains the code for running the experiments on the time-series classification benchmarks with all the reservoir methods (EuSN, ESN, R-ESN, DeepESN) used in the paper for the sequential MNIST task (the only difference is the usage of a buffering approach while computing the reservoir states, to keep the one-shot training of the readout)
* `experiments_trainable.py`, which contains all the code for running the experiments on the time-series classification benchmarks with all the fully trainable models (GRU, A-RNN, RNN, 1D-CNN) used in the paper
* `experiments_ts_modeling.py`, which contains the code for running the experiments on the time-series modeling benchmarks with EuSN and ESN, with and without input-readout connections, as used in the paper.

## Datasets

The pool of datasets used in the paper can be downloaded from the following link 
https://www.dropbox.com/sh/ewsym947w95fgjd/AAC9gnGIVLBjUXq9aYtfVkrea?dl=0
