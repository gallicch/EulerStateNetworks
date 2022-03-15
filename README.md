# EulerStateNetworks
This repository contains the TensorFlow 2.0 / Keras implementation of Euler State Networks (EuSN), as described in the paper
C. Gallicchio, "Euler State Networks", Submitted to Journal (2022)

## Files
Currently, two files are provided
* `euler.py`, which contains the main classes definition, including the EulerReservoirCell, the EuSN, and all the recurrent layers and architectures (both Reservoir Computing-based and fully trainable) used in the experiments
* `paper_experiments.py`, which provides the code for replicating the experiments on the time-series classification datasets reported in the paper.

## Datasets
The datasets used in this paper can be downloaded, individually for each task, using the function `load_task_data(task_name)` in `paper_experiments.py`, where `task_name` is a string that indicates one of the used tasks (i.e., 'Adiac', 'CharacterTrajectories', 'ECG5000', 'Epilepsy', 'Heartbeat', 'Libras', 'ShapesAll', 'Wafer', 'HandOutlines', 'IMDB_embedded', 'Reuters_embedded', 'SpokenArabicDigits'). 
In this case you need to import the gdown package (e.g., `!pip install gdown`). See an example of usage (including dataset download and experiment run with all the considered neural network architectures) in the function `run_all_experiments(task_name)` in `paper_experiments.py`.

Alternatively, the pool of datasets used in the paper can be downloaded from the following link https://www.dropbox.com/sh/ewsym947w95fgjd/AAC9gnGIVLBjUXq9aYtfVkrea?dl=0

