# Data_Dependent_Randomized_Smoothing
This is the official repo for the work "Data Dependent Randomized Smoothing"

Preprint: https://arxiv.org/pdf/2012.04351.pdf

## Environment Installations:
First, you need to install the environment from the provided yml file by running:

`conda env create -f ddsmoothing.yml`

Then, activate the envionment by running:

`conda activate ddsmoothing`

## Reproducing our Results:

For Cifar10 results, navigate to CIFAR10 directory by running `cd CIFAR10` and run the corresponding main file with the hyperparameters mentioned in the paper. For ImageNet results, the pretrained weights shall be downloaded from the original repo of the paper. Also, modify `all_datasets.py` for the path where ImageNet is saved.

## Using Data-Dependent Smoothing For other Frameworks:

The file `OptimzeSigma.py` contains the main function the optimizes for the sigmas. Embed this function in any randomized smoothing based training and enjoy a model with better certified robustness.

## Certifying the model with data dependent smoothing.

To Certify a model with data dependent randomized smoothing, we use the repo https://github.com/locuslab/smoothing where we replace `certify.py` with `certify_ds.py`. 
