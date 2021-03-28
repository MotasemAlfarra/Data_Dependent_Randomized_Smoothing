# Data Dependent Randomized Smoothing
This is the official repo for the work "Data Dependent Randomized Smoothing"

Preprint: https://arxiv.org/pdf/2012.04351.pdf

## Environment Installations:
First, you need to install the environment from the provided yml file by running:

`conda env create -f ddsmoothing.yml`

Then, activate the envionment by running:

`conda activate ddsmoothing`

## Reproducing our Results:

For Cifar10 results, navigate to CIFAR10 directory by running `cd CIFAR10` and run the corresponding main file with the hyperparameters mentioned in the paper. For ImageNet results, the pretrained weights shall be downloaded from the original repo of the corresponding paper. Also, modify `all_datasets.py` for the path where ImageNet is saved.

## Certifying the model with data dependent smoothing.

To Certify a model with data dependent randomized smoothing, we use the repo https://github.com/locuslab/smoothing where we replace `certify.py` with `certify_ds.py`. 

## Wanna Use DS in a New RS Training framework ?

No problem, all what you need is to use the code in `optimze_sigma.py` within the training fromwork. Upon training, run `optimze_sigma.py` for the samples in the test set with propper setting of the hyperparameters. Certify the final model with the output parameters gotten from the previous step.


@misc{alfarra2020data,

      title={Data Dependent Randomized Smoothing}, 
      
      author={Motasem Alfarra and Adel Bibi and Philip H. S. Torr and Bernard Ghanem},
      
      year={2020},
      
      eprint={2012.04351},
      
      archivePrefix={arXiv},
      
      primaryClass={cs.LG} 
      
}
