# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from models import resnet
from var_estimator_network import FCNetwork
import torch.nn as nn


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--fix-sig-smooth', action='store_true', default=False, help='certify with fixed sigma')
parser.add_argument("--path_sigma", type=str, help="path to sigma")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(path):
    model = resnet.resnet18(num_classes=10).to('cuda')
    checkpoint = torch.load(path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    print('Pretrained Model is loaded ! Go and certify now :)')
    return model


if __name__ == "__main__":
    # load the base classifier
    model = get_model(args.path)
    # create the smooothed classifier g
    # smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    if not args.fix_sig_smooth:
        sigma_test = torch.load(args.path_sigma)
    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tsigma\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
       
        #Smooth the classifier with this sigma
        if not args.fix_sig_smooth:
            args.sigma = sigma_test[i].item()
        print('sigma is: ', args.sigma)
        smoothed_classifier = Smooth(model, get_num_classes(args.dataset),args.sigma)
        #Now you can use the same exac
        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)
        print(radius)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{:.3}\t{}".format(
            i, label, prediction, radius, correct, args.sigma, time_elapsed), file=f, flush=True)

    f.close()
