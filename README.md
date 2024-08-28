# Conditional UDE
Repository accompanying the manuscript on conditional universal differential equations.

## What is a conditional UDE?
Whereas universal differential equations do not directly accomodate the derivation of model components from data containing variability between samples, such as measurements on individuals in a population, the conditional UDE accounts for this through additional learnable inputs. A cUDE is trained with a global neural network parameter set, and a set of conditional parameters that allow for explanation of the between-sample variability. This setup does then require a test set, where the neural network parameters are fixed and only the conditional parameters are estimated.

## Files in this repository
<!-- TODO: add explanation -->
