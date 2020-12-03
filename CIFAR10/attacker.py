import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from abc import ABCMeta, abstractmethod

class Attacker(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, inputs, targets):
        raise NotImplementedError
# Taken from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/
class PGD_L2(Attacker):
    """
    PGD attack
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    """

    def __init__(self,
                 steps: int,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(PGD_L2, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors=1, targeted: bool = False, no_grad=False) -> torch.Tensor:
        if num_noise_vectors == 1:
            return self._attack(model, inputs, labels, noise, targeted)
        else:
            if no_grad:
                with torch.no_grad():
                    return self._attack_mutlinoise_no_grad(model, inputs, labels, noise, num_noise_vectors, targeted)
            else:
                    return self._attack_mutlinoise(model, inputs, labels, noise, num_noise_vectors, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):
            adv = inputs + delta
            if noise is not None:
                adv = adv + noise
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return inputs + delta


    def _attack_mutlinoise(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):

            adv = inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = model(adv)

            pred_labels = logits.argmax(1).reshape(-1, num_noise_vectors).mode(1)[0]
            # safe softamx
            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(-1, num_noise_vectors, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, labels)
            
            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
       
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs[::num_noise_vectors])
            delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)


    def _attack_mutlinoise_no_grad(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1,targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):

            adv = inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = model(adv)

            pred_labels = logits.argmax(1).reshape(-1, num_noise_vectors).mode(1)[0]
            # safe softamx
            
            softmax = F.softmax(logits, dim=1)
            grad = F.nll_loss(softmax,  labels.unsqueeze(1).repeat(1,1,num_noise_vectors).view(batch_size*num_noise_vectors), 
                            reduction='none').repeat(*noise.shape[1:],1).permute(3,0,1,2)*noise
            
            grad = grad.reshape(-1, num_noise_vectors, *inputs.shape[1:]).mean(1)            
            # average the probabilities across noise

            grad_norms = grad.view(batch_size, -1).norm(p=2, dim=1)
            grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])

            # optimizer.step()
            delta = delta + grad*self.max_norm/self.steps*2

            delta.data.add_(inputs[::num_noise_vectors])
            delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)