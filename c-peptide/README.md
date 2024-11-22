# C-peptide Model
This folder contains all files for the c-peptide model included in the manuscript. Below is a description of each experiment and the other files in this folder

## Experiments

### Data Preparation
The file `00-prepare-data.jl` reads in the Ohashi and Fujita datasets and prepares the data for the model fitting process. The data is split into a training and test set, and individuals with missing data are removed. The data is then saved to be used in the model fitting process.

### Model fit with conventional UDE
The file `01-non-conditional.jl` fits the conventional UDE model to the Ohashi dataset.

### Model fit with conditional UDE
The file `02-conditional.jl` fits the UDE model with a conditional parameter to the Ohashi dataset. This file takes the training set from the Ohashi dataset and fits the selected UDE model to this training data. The model with the lowest validation loss is selected, and the neural network parameters are fixed. The conditional parameter is then estimated for all individuals in the test set from the Ohashi dataset. Additionally, the correlation analyses with clamp indices are performed. Finally, simulated data from the neural network is saved to be used with Symbolic Regression.

### Model Fit after Symbolic Regression
In the file `03-symreg.jl`, the best model from the symbolic regression run, based on the score metric and the loss, as defined in PySR was implemented and the $\beta$-parameter was re-estimated on the Ohashi dataset and the correlation analysis was repeated.

### Model fit on external data after symbolic regression
The file `04-symreg-external.jl` fits the found model with symbolic regression to the external data from the Fujita dataset[^1].

### Performance of the model with different data sizes
The file `05-performance-less-data.jl` fits the UDE model to fractions of the Ohashi train dataset with varying amounts of data and compares test performance of each fraction.

## Other Files

### `src/c-peptide-ude-models.jl`
This file contains general functions for the model fitting process that are used accross multiple experiments. 

[^1]: Fujita, S., Karasawa, Y., Hironaka, K. I., Taguchi, Y. H., & Kuroda, S. (2023). Features extracted using tensor decomposition reflect the biological features of the temporal patterns of human blood multimodal metabolome. PLoS ONE, 18(2 February). https://doi.org/10.1371/journal.pone.0281594