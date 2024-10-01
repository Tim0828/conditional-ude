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
This file performs the supplementary analysis of the model fit experiment, but with _two conditional parameters_ instead of one. 

### Model fit on external data after symbolic regression
The file `symreg-external-data.jl` fits the found model with symbolic regression to the external data from the Fujita dataset[^1].

### Model fit with non-conditional UDE
The file `model-fit-non-conditional.jl` fits the UDE model without the conditional parameter to the Ohashi dataset and the returned model fit is compared to the model fit with the conditional parameter.

### Performance of the model with different data sizes
The file `performance-less-data.jl` fits the UDE model to fractions of the Ohashi train dataset with varying amounts of data and compares test performance of each fraction.

## Other Files

### `models.jl`
This file contains general functions for the model fitting process that are used accross multiple experiments. 

[^1]: Fujita, S., Karasawa, Y., Hironaka, K. I., Taguchi, Y. H., & Kuroda, S. (2023). Features extracted using tensor decomposition reflect the biological features of the temporal patterns of human blood multimodal metabolome. PLoS ONE, 18(2 February). https://doi.org/10.1371/journal.pone.0281594