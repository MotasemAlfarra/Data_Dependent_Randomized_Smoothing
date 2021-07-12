import torch
from torch.autograd import Variable

from .certificate import Certificate


def optimize_isotropic_dds(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        sig_0: torch.Tensor, iterations: int, samples: int,
        device: str = 'cuda:0'
) -> torch.Tensor:
    """Optimize smoothing parameters for a batch.

    Args:
        model (torch.nn.Module): trained network
        batch (torch.Tensor): inputs to certify around
        certificate (Certificate): instance of desired certification object
        learning_rate (float): optimization learning rate for ANCER
        sig_0 (torch.Tensor): initialization value per input in batch
        iterations (int): number of iterations to run the optimization
        samples (int): number of samples per input and iteration
        device (str, optional): device on which to perform the computations

    Returns:
        torch.Tensor: optimized isotropic thetas
    """

    batch_size = batch.shape[0]

    sig = Variable(sig_0, requires_grad=True).view(batch_size, 1, 1, 1)

    for param in model.parameters():
        param.requires_grad_(False)

    # Reshaping so for n > 1
    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    for _ in range(iterations):
        sigma_repeated = sig.repeat((1, samples, 1, 1)).view(-1, 1, 1, 1)
        eps = certificate.sample_noise(
            new_batch, sigma_repeated)  # Reparamitrization trick
        out = model(new_batch + eps).reshape(
            batch_size, samples, - 1).mean(1)  # This is \psi in the algorithm

        vals, _ = torch.topk(out, 2)
        radius = certificate.compute_radius_estimate(vals, sig.reshape(-1))
        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += learning_rate*grad[0]  # Gradient Ascent step

    # For training purposes after getting the sigma
    for param in model.parameters():
        param.requires_grad_(True)

    return sig.reshape(-1)
