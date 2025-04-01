# Bachelor Thesis
This repository was extended as part of my bachelor thesis at the DSME (RWTH-Aachen), titled 'Exploring Last-Layer Surrogate Models for Model-based Reinforcement Learning'.

## Extended Functionality
The primary extensions introduced in this work are listed below. It should be noted that additional modifications were made to various utility and helper functions, as well as core files, which are not explicitly listed here.

- Implementation of a cache of the noise and W distribution for many predictions without retraining inbetween [link](vbll/layers/regression.py)
- Implementation of a fixed initialization to improve stability at the beginning of training [link](vbll/layers/regression.py)
- Ability to create Thompson heads tosample MLPs from the VBLL model [link](vbll/layers/regression.py)

- Comparison of different models and methods within one model on test data
    - Currently supported models: [MLP](model_comparison/models/mlp.py), [PNN](model_comparison/models/pnn.py), [PE](default_gaussian_mean_var_ensemble), [VBLL](model_comparison/models/vbll_mlp.py) (also available as ensemble)
- Mehtods available for comparison:
    - Thompson sampling 
    - Recursive updates
    - Initialization
    - Probabilistic ensemble predicting strategy

### Usage
There is a README file containing the usage information in the comparison folder [link](model_comparison/README.md).