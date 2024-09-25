# C-peptide Model
This folder contains all files for the c-peptide model included in the manuscript. Below is a description of each experiment and the other files in this folder

## Experiments

### Selecting the Neural Network Size
This experiment is included in the file `model-selection.jl`. In this file, we take the training set from the Ohashi dataset, and split it into a training set, and a validation set. We use the training set to fit neural networks of varying width and depth and evaluate the resulting neural networks by performing a model fit of the conditional parameter on the validation split. We selected the model with the lowest mean validation error, which is the model with 2 layers of size 6.

### Model Fit
This is the main experiment in the manuscript, which is included in `model-fit.jl`. This file takes the training set from the Ohashi dataset and fits the selected UDE model to this training data. The model with the lowest training loss is selected, and the neural network parameters are fixed. The conditional parameter is then estimated for all individuals in the test set from the Ohashi dataset. Additionally, the correlation analyses with clamp indices are performed. Finally, simulated data from the neural network is saved to be used with Symbolic Regression.

### Model Fit after Symbolic Regression
The best model from the symbolic regression run, based on the score metric and the loss, as defined in PySR was implemented and the $\beta$-parameter was re-estimated on the Ohashi dataset and the correlation analysis was repeated.

### Other Indices
The file `other-indices.jl` computes existing indices of $\beta$-cell function based on the OGTT data and performs a correlation analysis with clamp indices. 

### Two Conditional Parameters
This file performs the supplementary analysis of the model fit experiment, but with _two conditional parameters_ instead of one. <!-- TODO: Add additional information about this experiment -->

## Other Files

### `models.jl`
This file contains general functions for the model fitting process that are used accross multiple experiments. 