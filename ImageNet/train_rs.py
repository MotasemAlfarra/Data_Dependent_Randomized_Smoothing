import argparse
import torch
from torch import nn

import all_datasets
# from models import resnet
# from var_estimator_network import FCNetwork
from torchvision.models.resnet import resnet50
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from os import path
# from expectation import get_gaussian_expectation
from utils import copy_pretrained_model, imagenet_wrapper, compute_loss, save_checkpoint, plot_samples, macer_loss, load_optimizer
import numpy as np

torch.manual_seed(0)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# print(device)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


print('Using torch version {}'.format(torch.__version__))
print('Using {} device'.format(device))

# Training settings
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset-path', type=str, default='/local/reference/CV/ILSVR/classification-localization/data/jpeg',
                    help='path to imagenet dataset')
parser.add_argument('--output_path', default='.',
                    help='path to save exps')
parser.add_argument('--batch_sz', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=440, metavar='N',
                    help='number of epochs to train (default: 90)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--momentum', type=int, default=0.9,
                    help='momentum for optimizer')
parser.add_argument('--weight_decay', type=int, default=5e-4,
                    help='weight decay for optimizer')
parser.add_argument('--step_sz', type=int, default=30,
                    help='learning rate drop every step_sz epochs for optimizer')
parser.add_argument('--gamma', type=int, default=0.1,
                    help='gamma factor to drop learning rate for optimizer')
#DS arguments
parser.add_argument('--sig0_tr', type=float, default=0.12,
                    help='initial sigma to start with in training')
parser.add_argument('--sig0_ts', type=float, default=0.12,
                    help='initial sigma to start with in testing')
parser.add_argument('--lr_sig', type=float, default=0.0001,
                    help='learning rate of optimizing sigma')
parser.add_argument('--iter_sig_tr', type=int, default=1,
                    help='epochs to backprob through sigma during training')
parser.add_argument('--iter_sig_ts', type=int, default=0,
                    help='epochs to backprob through sigma during testing')
parser.add_argument('--iter_sig_after', type=int, default=100,
                    help='epochs to backprob through sigma during testing')
parser.add_argument('--epoch-switch', type=int, default=0,
                    help='epoch at which we will start optimizing for sigma')
parser.add_argument('--gaussian_num_ds', type=int, default=1, metavar='N',
                    help='number of gaussian samples per instance for the data smoothing')
#MACER arguments
parser.add_argument('--gaussian_num', type=int, default=1, metavar='N',
                    help='number of gaussian samples per instance for the macer loss')
parser.add_argument('--lbd', type=float, default=0.0, help='Macer lambda: weight of robustness loss')
parser.add_argument('--gam', type=float, default=0.0, help='Macer gamma: Hinge factor')
#checkpoint to start with or only compute sigma for the test set
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint to start with')
parser.add_argument('--path-sigma', type=str, default=None, help='path to sigma for the test set to start with')
parser.add_argument('--center-only', action='store_true', default=False, help='forward pass with only subtracting the mean')

args = parser.parse_args()


train_loader, test_loader, img_sz, trainset_size, testset_size = all_datasets.ImageNet(args.dataset_path, args.batch_sz)


def get_sigma(model, batch, lr_sig, sig_0, iters, device='cuda', ret_radius = False, gaussian_num=1):
    sig = torch.autograd.Variable(sig_0, requires_grad=True).view(batch.shape[0], 1, 1, 1)
    # mean, std = torch.zeros(batch.shape[0]), torch.ones(batch.shape[0])
    mean, std = batch.new_zeros(batch.shape[0]), batch.new_ones(batch.shape[0])
    m = torch.distributions.normal.Normal(mean, std)
    # model.eval()
    batch_size = batch.shape[0]
    for param in model.parameters():
        param.requires_grad_(False)
    
    # new_shape = [batch_size * gaussian_num]
    # new_shape.extend(batch[0].shape)
    # new_batch = batch.repeat((1,gaussian_num, 1, 1)).view(new_shape)
    # sigma_repeated = sig.repeat((1, gaussian_num, 1, 1)).view(-1,1,1,1)

    for _ in range(iters):
      
        eps = torch.randn_like(batch)*sig
        out = model(batch + eps).reshape(batch_size, gaussian_num, 1000).mean(1)
        
        vals, ind = torch.topk(out, 2)
        vals.transpose_(0, 1)
        gap = m.icdf(vals[0].clamp_(0.02, 0.98)) - m.icdf(vals[1].clamp_(0.02, 0.98))
        radius = sig.reshape(-1)/2 * gap  # The radius formula
        grad = torch.autograd.grad(radius.sum(), sig)
        sig.data += lr_sig*grad[0]  # Gradient Ascent step

    for param in model.parameters():
        param.requires_grad_(True)    
    # model.train()
    eps = torch.randn_like(batch)*sig
    if ret_radius:
        return sig.reshape(-1), batch + eps, radius
    return sig.reshape(-1), batch + eps




def train(epoch, model, train_loader, optimizer, writer,
          sigma_0, lr_sigma, iters_sig, gaussian_num=1, lamda=0.0, gamma=0.0, gaussian_num_ds=1):
    model = model.train()
    train_loss = 0
    total = 0
    correct = 0
    # CE_loss = nn.CrossEntropyLoss()
    import time
    start_time = time.time()
    for batch_idx, (batch, targets, idx) in enumerate(train_loader):
        start_time = time.time()
        optimizer.zero_grad()

        batch_size = len(idx)
        batch = batch.to(device)
        targets = targets.to(device)
        
        # model.eval()
        sigma, batch_corrupted = get_sigma(model, batch, lr_sigma, sigma_0[idx], iters_sig, device, gaussian_num=gaussian_num_ds)
        # model.train()
        sigma_0[idx] = sigma  # updating sigma

        # #repeating the input for computing the macer loss
        # new_shape = [batch_size * gaussian_num]
        # new_shape.extend(batch[0].shape)
        # batch = batch.repeat((1,gaussian_num, 1, 1)).view(new_shape)
        # #repeating sigmas to do the monte carlo
        # sigma_repeated = sigma.repeat((1, gaussian_num, 1, 1)).view(-1,1,1,1)
        # noise = torch.randn_like(batch)*sigma_repeated

        # batch_corrupted = batch + noise

        outputs_softmax = model(batch_corrupted).reshape(batch_size, gaussian_num, 1000).mean(1)#1000 here is for ImageNet
        # clean_output = model(batch)
        
        total_loss = compute_loss(outputs_softmax, targets)
        if torch.isnan(outputs_softmax).any() or torch.isnan(total_loss).any():
            print('NaN flag')
        # total_loss += lamda*macer_loss(outputs_softmax, targets, sigma, gamma) 
        # clean_loss = compute_loss(clean_output, targets)
        # total_loss += clean_loss

        train_loss += total_loss.item()*len(batch)
        _, predicted = outputs_softmax.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # update parameters
        total_loss.backward()
        optimizer.step()
        print('Required time (mins) for a batch is: ',(time.time()-start_time)/60.0)
        
        if batch_idx % 100 == 0:
            
            print('+ Epoch: {}. Iter: [{}/{} ({:.0f}%)]. Loss: {}. Accuracy: {}'.format(
                    epoch,
                    batch_idx * len(batch),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    train_loss / total,
                    100.*correct / total
                    ))

    n = min(batch.size(0), 8)
    comparison = torch.cat([batch[:n], batch_corrupted[:n]])
    comparison = torch.clamp(comparison, min=0, max=1)
    fig = plot_samples(comparison.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(), h=2, w=n)

    writer.add_figure('sample of noisy trained examples', fig, epoch)
    writer.add_scalar('loss/train_loss', train_loss / total, epoch)
    writer.add_scalar('accuracy/train_accuracy', 100.*correct / total, epoch)
    writer.add_scalar('sigma/train_sigma_mean', sigma_0.mean().item(), epoch)
    writer.add_scalar('sigma/train_sigma_min', sigma_0.min().item(), epoch)
    writer.add_scalar('sigma/train_sigma_max', sigma_0.max().item(), epoch)

    return sigma_0


def test(epoch, model, test_loader, writer, sigma_0, lr_sigma, iters_sig):
    model = model.eval()
    test_loss = 0
    test_loss_corrupted = 0
    total = 0
    correct = 0
    correct_corrupted = 0
    for _, (batch, targets, idx) in enumerate(test_loader):
        batch = batch.to(device)
        targets = targets.to(device)

        sigma, batch_corrupted = get_sigma(model, batch, lr_sigma, sigma_0[idx], iters_sig, device)
        sigma_0[idx] = sigma  # update sigma
        with torch.no_grad():

            # forward pass through the base classifier
            outputs_softmax = model(batch)
            outputs_corrputed_softmax = model(batch_corrupted)


        loss = compute_loss(outputs_softmax, targets)
        loss_corrupted = compute_loss(outputs_corrputed_softmax, targets)

        test_loss += loss.item()*len(batch)
        test_loss_corrupted += loss_corrupted.item()*len(batch)

        _, predicted = outputs_softmax.max(1)
        _, predicted_corrupted = outputs_corrputed_softmax.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        correct_corrupted += predicted_corrupted.eq(targets).sum().item()

    print('===> Test Loss: {}. Test Accuracy: {}. Test Loss Corrupted: {}. Test Accuracy Corrupted: {}'.format(
                    test_loss / total,
                    100.*correct / total,
                    test_loss_corrupted / total,
                    100.*correct_corrupted / total
                    ))
    n = min(batch.size(0), 8)
    comparison = torch.cat([batch[:n], batch_corrupted[:n]])
    comparison = torch.clamp(comparison, min=0, max=1)
    fig = plot_samples(comparison.detach().cpu().numpy().transpose(0,2,3,1).squeeze(), h=2, w=n)

    writer.add_figure('sample of noisy test examples', fig, epoch)
    writer.add_scalar('loss/test_loss', test_loss / total, epoch)
    writer.add_scalar('accuracy/test_accuracy', 100.*correct / total, epoch)
    writer.add_scalar('loss/test_loss_corrupted', test_loss_corrupted / total, epoch)
    writer.add_scalar('accuracy/test_accuracy_corrupted', 100.*correct_corrupted / total, epoch)
    writer.add_scalar('sigma/test_sigma_mean', sigma_0.mean().item(), epoch)
    writer.add_scalar('sigma/test_sigma_min', sigma_0.min().item(), epoch)
    writer.add_scalar('sigma/test_sigma_max', sigma_0.max().item(), epoch)

    return 100.*correct_corrupted / total, sigma_0




def optimize_sigma(model, loader, writer, sigma_0, lr_sigma, flag='train', radius = None, gaussian_num_ds=1, epoch=1):
    model = model.eval()
    total = 0
    test_loss, test_loss_corrupted = 0, 0
    correct, correct_corrupted= 0, 0

    
    for _, (batch, targets, idx) in enumerate(loader):
        batch, targets = batch.to(device), targets.to(device) #Here I will put iters to 1 as the outer loop contains the number of iterations
        sigma, batch_corrupted, rad = get_sigma(model, batch, lr_sigma, sigma_0[idx], 1, device,
                                                    ret_radius=True, gaussian_num=gaussian_num_ds)
        sigma_0[idx], radius[idx] = sigma, rad

        with torch.no_grad():
            outputs_corrputed_softmax = model(batch_corrupted)

        _, predicted_corrupted = outputs_corrputed_softmax.max(1)
        total += targets.size(0)
        correct_corrupted += predicted_corrupted.eq(targets).sum().item()
    #plottings
    n = min(batch.size(0), 8)
    comparison = torch.cat([batch[:n], batch_corrupted[:n]])
    comparison = torch.clamp(comparison, min=0, max=1)
    fig = plot_samples(comparison.detach().cpu().numpy().transpose(0,2,3,1).squeeze(), h=2, w=n)

    writer.add_figure('optimizing sigma sample of noisy '+flag+' examples', fig, epoch)
    # writer.add_scalar('optimizing_sigma/'+flag+'/loss_clean', test_loss / total, epoch)
    # writer.add_scalar('optimizing_sigma/'+flag+'/accuracy_clean', 100.*correct / total, epoch)
    # writer.add_scalar('optimizing_sigma/'+flag+'/loss_corrupted', test_loss_corrupted / total, epoch)
    writer.add_scalar('optimizing_sigma/'+flag+'/accuracy_corrupted', 100.*correct_corrupted / total, epoch)
    writer.add_scalar('optimizing_sigma/'+flag+'/sigma_mean', sigma_0.mean().item(), epoch)
    writer.add_scalar('optimizing_sigma/'+flag+'/sigma_min', sigma_0.min().item(), epoch)
    writer.add_scalar('optimizing_sigma/'+flag+'/sigma_max', sigma_0.max().item(), epoch)
    writer.add_scalar('optimizing_sigma/'+flag+'/radius_for_sample_0', radius[0].item(), epoch)
    #Saving the sigmas
    return sigma_0



def main():
    plt.ioff()

    model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
    if args.checkpoint is not None:
        model = copy_pretrained_model(model, args.checkpoint)
    if args.center_only:    
        model = imagenet_wrapper(model, std = [1.0, 1.0, 1.0]) #to do the normalization inside the forward pass
    else:
        model = imagenet_wrapper(model)

    radius_train = torch.zeros(trainset_size)
    radius_test = torch.zeros(testset_size)
    sigma_train = torch.ones(trainset_size)*args.sig0_tr
    sigma_test = torch.ones(testset_size)*args.sig0_ts

    radius_train, radius_test = radius_train.to(device), radius_test.to(device)
    sigma_train, sigma_test = sigma_train.to(device), sigma_test.to(device)

    if args.path_sigma is not None: #loading test sigma to optimize for it
        sigma_test = torch.load(args.path_sigma)

    base_bath_save = f'scratch_exps/ImageNet/exps/{args.output_path}/sig0_tr_{args.sig0_tr}/sig0_ts_{args.sig0_tr}/lr_sig_{args.lr_sig}/iter_sig_tr_{args.iter_sig_tr}/iter_sig_ts_{args.iter_sig_ts}'
    tensorboard_path = f'scratch_exps/ImageNet/tensorboard/{args.output_path}/sig0_tr_{args.sig0_tr}/sig0_ts_{args.sig0_tr}/lr_sig_{args.lr_sig}/iter_sig_tr_{args.iter_sig_tr}/iter_sig_ts_{args.iter_sig_ts}'

    if not path.exists(base_bath_save):
        os.makedirs(base_bath_save)

    writer = SummaryWriter(tensorboard_path, flush_secs=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_sz, gamma=args.gamma)

    if args.checkpoint is not None:
        optimizer = load_optimizer(optimizer, args.checkpoint)
    
    best_acc = 0.0
    for epoch in range(args.epochs):

        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch >= args.epoch_switch:
            print('Switched to DS training, sigma train is {} and sigma test is {}'.format(sigma_train.mean().item(), sigma_test.mean().item()))
            sigma_train = train(epoch, model, train_loader, optimizer, writer, sigma_train, args.lr_sig, args.iter_sig_tr,
                                gaussian_num=args.gaussian_num, lamda=args.lbd, gamma=args.gam, gaussian_num_ds=args.gaussian_num_ds )
            test_acc, sigma_test = test(epoch, model, test_loader, writer, sigma_test, args.lr_sig, args.iter_sig_ts)
        else:
            print('Training with RS')
            sigma_train = train(epoch, model, train_loader, optimizer, writer, sigma_train, args.lr_sig, 0,
                                gaussian_num=args.gaussian_num, lamda=args.lbd, gamma=args.gam )
            test_acc, sigma_test = test(epoch, model, test_loader, writer, sigma_test, args.lr_sig, 0)

        scheduler.step()
        print('Learing Rate: {}.'.format(optimizer.param_groups[0]['lr']))

        if test_acc > best_acc:
            best_flag = True
            best_acc = test_acc
        else:
            best_flag = False

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_param': optimizer.state_dict(),
        }, base_bath_save, best=best_flag)
    
    #Optimizing sigma afterwards
    print('Training has finished now we optimize for the sigmas. So far, sigma train is {} and sigma test is {}'.format(sigma_train.mean().item(), sigma_test.mean().item()))
    for i in range(1, args.iter_sig_after+1):
        print('Optimizing sigmas for the iteration {} out of {}'.format(i, args.iter_sig_after))
        sigma_test = optimize_sigma(model, test_loader, writer, sigma_test, args.lr_sig,
                                    flag='test', radius = radius_test, gaussian_num_ds = args.gaussian_num_ds, epoch=i)
        # Saving the sigmas
        torch.save(sigma_test, base_bath_save + '/sigma_test_' + str(i) + '.pth')
    print('everything is done, you should be happy now !')
    
# python train_model.py --dataset cifar10 --arch resnet18
if __name__ == "__main__":
    main()
    
# check_inf = torch.isinf(gap)
# if check_inf.any():
#     print('vals0 are ', vals[0][check_inf])
#     print('vals1 are ', vals[1][check_inf])
#     print('sigma is ', sig[check_inf])
#     print('gap is: ',  gap[check_inf])
            # if torch.isnan(sigma).any():
    #     print('sigma nan')
    #     if torch.isnan(batch_corrupted).any():
    #         print('batch_corrupted nan')
    #         if torch.isnan(outputs_corrputed_softmax).any():
    #             print('outputs_corrputed_softmax nan')
