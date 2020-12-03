import argparse
import torch
from torch import nn

# from var_estimator_network import FCNetwork
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from os import path
# from expectation import get_gaussian_expectation

import numpy as np

torch.manual_seed(0)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def copy_pretrained_model(model, path_to_copy_from):
    resnet = torch.load(path_to_copy_from, map_location='cuda')
    print(resnet.keys())
    if 'state_dict' in resnet.keys(): #For RS and ARS
        resnet = resnet['state_dict']
    if 'net' in resnet.keys(): #For MACER guys
        resnet = resnet['net']
    keys = list(resnet.keys())
    # print(keys)
    # print(resnet['fc.bias'].shape)
    count = 0
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(resnet[keys[count]].data)
        count +=1
    
    print('Pretrained model is loaded successfully')
    return model

def load_optimizer(optimizer, path):
    checkpoint = torch.load(path, map_location='cuda')
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('optimizer is successfully loaded')
    return optimizer

class imagenet_wrapper(nn.Module):
    def __init__(self, model, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(imagenet_wrapper, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.mean = torch.tensor(mean).view(1,3,1,1).to('cuda')
        self.std = torch.tensor(std).view(1,3,1,1).to('cuda')
    def forward(self, x):
        out = self.model((x-self.mean)/self.std)
        return self.softmax(out)


def compute_loss(outputs_softmax, targets):
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    if torch.isnan(outputs_logsoftmax).any():
            print('Fuck soft max')
    loss = nn.NLLLoss()
    classification_loss = loss(outputs_logsoftmax, targets)
    return classification_loss


def macer_loss(probs, labels, sigma, gamma = 0.1, device = 'cuda'):
    vals, ind = torch.topk(probs, 2)
    correct_ind = ind[:, 0] == labels
    m = torch.distributions.normal.Normal(torch.zeros(sum(correct_ind).item()).to(device),
                                          torch.ones(sum(correct_ind).item()).to(device))
    gap = m.icdf(vals[correct_ind, 0].clamp_(0.02, 0.98)) - m.icdf(vals[correct_ind, 1].clamp_(0.02, 0.98))
    radius = sigma[correct_ind].reshape(-1)/2 * gap.to(device)
    return torch.relu(gamma - radius).mean()


def save_checkpoint(state, base_bath_save, best = False):
    torch.save(state, base_bath_save + '/checkpoint.pth.tar')
    if best:
        torch.save(state, base_bath_save + '/best_model.pth.tar')


def plot_samples(samples, h=5, w=10):
    plt.ioff()
    fig, axes = plt.subplots(nrows=h,
                             ncols=w,
                             figsize=(int(1.4 * w), int(1.4 * h)),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap='gray')
    plt.close(fig)
    return fig
