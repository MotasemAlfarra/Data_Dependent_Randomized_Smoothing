# Data Dependent Randomized Smoothing
This is the official repo for the work "Data Dependent Randomized Smoothing"

This work was accepted to the 38th Conference on Uncertainity in AI (UAI 2022)

Paper: https://proceedings.mlr.press/v180/alfarra22a.html

![plot](./pull.png)

## Environment Installations:
First, you need to install the environment from the provided yml file by running:

`conda env create -f ddsmoothing.yml`

Then, activate the envionment by running:

`conda activate ddsmoothing`

## Reproducing our Results:

For Cifar10 results, navigate to CIFAR10 directory by running `cd CIFAR10` and run the corresponding main file with the hyperparameters mentioned in the paper. For ImageNet results, the pretrained weights shall be downloaded from the original repo of the corresponding paper. Also, modify `all_datasets.py` for the path where ImageNet is saved. Alternatively, all pretrained models can be found [here](https://drive.google.com/drive/folders/1t4IGRDmQ_qA8UVhkJYa3RWXFM6v-5vm5?usp=sharing).

## Certifying the model with data dependent smoothing.

To Certify a model with data dependent randomized smoothing, we use the repo https://github.com/locuslab/smoothing where we replace `certify.py` with `certify_ds.py`. 

## Wanna Use DS in a New RS Training framework ?

No problem, all what you need is to use the code in `optimze_sigma.py` within the training fromwork. Upon training, run `optimze_sigma.py` for the samples in the test set with propper setting of the hyperparameters. Certify the final model with the output parameters gotten from the previous step. Alternatively, you can install our package that contains the code for the optimization and certification by running:

`pip install ddsmoothing-python`

Next, import the package in your python code by running"

`import ddsmoothing`

Then import the opimization class `OptimizeIsotropicSmoothingParameters` from our package in your python code. This class takes a PyTorch model, a DataLoader and a device. 

`def __init__(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: str = "cuda:0"):`


Next, you need to define the `Certificate` class (either `L2Certificate` or `L1Certificate` depending on the norm of interest). You need to pass the batch size to this class.

`def __init__(self, batch_size: int, device: str = "cuda:0"):`

Finally, you can run the optimization by calling the `run_optimization` method from the `OptimizeIsotropicSmoothingParameters` class.

`def run_optimization(self, certificate: Certificate, lr: float, theta_0: torch.Tensor, iterations: int, num_samples: int, filename: str = './'):`

where:


certificate (Certificate): instance of desired certification object.

lr (float, optional): optimization learning rate for Isotropic DDS.

theta_0 (torch.Tensor): initialization value per input of the testloader.

iterations (int): Number of iterations for the optimization.

num_samples (int): number of samples per input and iteration.

filename (str, optional): name of the file of the saved thetas.


For further details, please check the examples in `ddsmoothing/scripts`.

## Citation

If you use this repo, please cite us:
```
@InProceedings{pmlr-v180-alfarra22a,
  title = 	 {Data dependent randomized smoothing},
  author =       {Alfarra, Motasem and Bibi, Adel and Torr, Philip H. S. and Ghanem, Bernard},
  booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {64--74},
  year = 	 {2022},
  editor = 	 {Cussens, James and Zhang, Kun},
  volume = 	 {180},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {01--05 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v180/alfarra22a/alfarra22a.pdf},
  url = 	 {https://proceedings.mlr.press/v180/alfarra22a.html}
}
```

This repository is licensed under the terms of the MIT license.
