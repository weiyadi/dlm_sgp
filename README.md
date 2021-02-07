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

###Conjugate
A simple sine dataset is created with 1000 data points for training and 200 for testing. 
`--seed` sets the random seed;
`--inducing` sets the number of inducing inputs;
`--method` can be 
set to `svgp`, `fitc`, `fixed-log-dlm`, `joint-log-dlm`, `fixed-sq-dlm`, `joint-sq-dlm`;
`--reg` sets the regularization parameter for KL-regularizer;
`--num_samples` sets the number of Monte Carlo samples if we want to use biased estimates and is only available for 
`fixed-log-dlm` and `joint-log-dlm`, when it is set 0, exact computation is used.
Below is an example:
```console
python ./conjugate/script.py --seed 0 --inducing 20 --method joint-log-dlm --reg 1.0 --num_samples 10
```

###Nonconjugate
A simple cosine dataset is created with 1000 data points for training and 200 for testing.
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
`--num_samples` sets the number of samples if we want to use biased estimates and is available for all dlm methods.
`--jitter` is set for smooth-bMC in the paper, 0 means no jitter added.
Below is an example.
```console
python ./nonconjugate/script.py --seed 0 --inducing 20 --likelihood binary --kern rbf --method joint-dlm \
--reg 1.0 --num_samples 10 --jitter 1e-4
```
An example script to collect bias statistics for bMC, smooth-bMC and uPS is also included. All gradients are collected
based on the toy data with binary likelihood. Notice that uPS is slow so it may take more than 40min to finish.
```console
python ./nonconjugate/collect_gradient_demo.py --method bMC
```
And the results are saved in the format of `condc-bMC-m.pdf`, `condc-bMC-chol.pdf`, `condd-bMC-m.pdf`, 
`condd-bMC-chol.pdf`, `err-bMC-m.pdf`, `err-bMC-chol.pdf`. For different methods, the middle part is changed to the
corresponding method name.
