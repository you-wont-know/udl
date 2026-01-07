# UDL Mini-Project

This repository contains code the UDL Mini-Project - reproducing and extending [this paper](https://proceedings.mlr.press/v70/gal17a). It reproduces sections 5.1., 5.2., and extends with Baysian Variational inference.

- **main.py**: contains parameter parsing, setup and logging.
- **acquisition_functions.py**: contains all acquisition functions implemented, which were used in the paper, and a few more.
- **cnn.py** contains the implementation of the CNN that was reproduced, as well as two CNN implementations for AI and MFVI CNNs. This file also contains the ELBO defined for the MFVI CNN.
- **data.py**: contains MNIST data splitting.
- **episode.py**: contains an implementation for a single episode. This file got a bit complicated to follow after adding the extensions (we wanted to reuse code).
- 

**Candidate number:** 1102191.

## Running the experiments
To run the reproduction experiments, run `bash run_reproducing.sh`. To run the extension experiments, run `bash run_extension.sh`.
