import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal
def OptimzeSigma(model, batch, alpha, sig_0, K, n):
    device='cuda:0'
    batch_size = batch.shape[0]

    sig = Variable(sig_0, requires_grad=True).view(batch_size, 1, 1, 1)
    m = Normal(torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device))

    for param in model.parameters():
        param.requires_grad_(False)

    #Reshaping so for n > 1
    new_shape = [batch_size * n]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1,n, 1, 1)).view(new_shape)

    for _ in range(K):
        sigma_repeated = sig.repeat((1, n, 1, 1)).view(-1,1,1,1)
        eps = torch.randn_like(new_batch)*sigma_repeated #Reparamitrization trick
        out = model(new_batch + eps).reshape(batch_size, n, 10).mean(1)#This is \psi in the algorithm
        
        vals, _ = torch.topk(out, 2)
        vals.transpose_(0, 1)
        gap = m.icdf(vals[0].clamp_(0.02, 0.98)) - m.icdf(vals[1].clamp_(0.02, 0.98))
        radius = sig.reshape(-1)/2 * gap  # The radius formula
        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += alpha*grad[0]  # Gradient Ascent step

    #For training purposes after getting the sigma
    for param in model.parameters():
        param.requires_grad_(True)    

    return sig.reshape(-1)
