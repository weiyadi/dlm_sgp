# Direct Loss Minimization for Sparse Gaussian Processes
This repository includes the basic Python code for the following paper:

Y. Wei, R. Sheth, and R. Khardon. Direct loss minimization for sparse gaussian processes.
https://arxiv.org/abs/2004.03083

Experiments in above paper require expensive computation and are done in a parallel way. 
This repository only provides basic usage on toy datasets.

Thank [Weizhe Chen](https://github.com/Weizhe-Chen) for contributing the basic structure of conjugate cases.

## Dependencies

 * [PyTorch](https://pytorch.org/)
 * [GPyTorch](https://gpytorch.ai/)

## Directory structure

This repository has the following directory structure
 * *README*: This file.
 * *conjugate*: Folder for conjugate sparse Gaussian processes, i.e. Gaussian likelihood.
 * *nonconjugate*: Folder for nonconjugate sparse Gaussian processes, including binary and poisson likelihood. 
 This folder also includes product sampling implementation.


## Training a model using different methods

### Demonstration of Conjugate Cases
The file `conjugate/script.py` shows how to apply different algorithms on a dataset. We provide a toy sine dataset for 
demonstration. And you can implement your own `load_data` method to use your own dataset.
`--dataset` sets the dataset;
`--n_train` sets the training size, default 1000;
`--n_test` sets the test size, default 200;
`--seed` sets the random seed;
`--inducing` sets the number of inducing inputs;
`--method` can be 
set to `svgp`, `fitc`, `fixed-log-dlm`, `joint-log-dlm`, `fixed-sq-dlm`, `joint-sq-dlm`;
`--reg` sets the regularization parameter for KL-regularizer;
`--auto_select_reg` is set when we want to automatically use validate set to select regularization;
`--num_samples` sets the number of Monte Carlo samples if we want to use biased estimates and is only available for 
`fixed-log-dlm` and `joint-log-dlm`, when it is set 0, exact computation is used.
Below is an example for fixed regularization:
```console
python ./conjugate/script.py --dataset sine --seed 0 --inducing 20 --method joint-log-dlm --reg 1.0 --num_samples 10
```
and an example for automatically selecting regularization:
```console
python ./conjugate/script.py --dataset sine --n_train 100 --seed 0 --inducing 20 --method joint-log-dlm \
--auto_select_reg
```
For fixed regularization, the training loss during optimization is printed; 
For automatically selecting regularization, performance on validate dataset and the best regularization parameter is printed.
For both cases, the performance on test dataset is printed.

### Demonstration of Nonconjugate Cases
A simple cosine dataset is created for demonstration. You can implement the `load_dataset` in 
`./nonconjugate/script.py`.
`--dataset` sets the data we would like to use, default is the 'toy' which uses the toy cosine dataset;
`--n_train` sets the number of training examples;
`--n_test` sets the number of testing examples;
`--seed` sets the random seed;
`--inducing` sets the number of inducing inputs;
`--likelihood` sets the likelihood type, i.e. `binary`, `poisson_exp`(Poisson likelihood with log link function) or 
`poisson_log1p`(Poisson likelihood with log1p link function);
`--mean_type` sets the mean function, i.e. `constant` or `zero`;
`--kern` sets the kernel type, `rbf` or `matern`.
`--method` can be 
set to `svgp`, `fixed-dlm`(exact computation if available, otherwise, Monte Carlo estimation), 
`joint-dlm`, `fixed-dlm-ps`(product sampling), `joint-dlm-ps`;
`--reg` sets the regularization parameter for KL-regularizer;
`--auto_select_reg` is set when we want to automatically use validate set to select regularization;
`--num_samples` sets the number of samples if we want to use biased estimates and is available for all dlm methods.
`--jitter` is set for smooth-bMC in the paper, 0 means no jitter added.
Below is an example for fixed regularization:
```console
python ./nonconjugate/script.py --seed 0 --inducing 20 --likelihood binary --kern rbf --method joint-dlm \
--reg 1.0 --num_samples 10 --jitter 1e-4
```
and an example for automatically selecting the regularization. Notice that dlm with product sampling is slow so we do 
not recommend using `fixed-dlm-ps` and `joint-dlm-ps`.
```console
python ./nonconjugate/script.py --seed 0 --inducing 20 --likelihood binary --kern rbf --method fixed-dlm \
--num_samples 10 --jitter 1e-4 --auto_select_reg
```

For fixed regularization, the training loss during optimization is printed; 
For automatically selecting regularization, performance on validate dataset and the best regularization parameter is printed.
For both cases, the performance on test dataset is printed.

### Bias Statistics
An example script to collect bias statistics for bMC, smooth-bMC and uPS is also included.
Please see the paper for interpretation of the plots and their relation to conditions for convergence.
All gradients are collected
based on the toy data with binary likelihood. Notice that uPS is slow so it may take more than 40min to finish.
```console
python ./nonconjugate/collect_gradient_demo.py --method bMC
```
And the results are saved in the format of `condc-bMC-m.pdf`, `condc-bMC-chol.pdf`, `condd-bMC-m.pdf`, 
`condd-bMC-chol.pdf`, `err-bMC-m.pdf`, `err-bMC-chol.pdf`. For different methods, the middle part is changed to the
corresponding method name.
