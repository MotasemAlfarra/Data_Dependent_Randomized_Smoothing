import argparse
from time import time
import torch
import datetime
from tqdm import tqdm

from ddsmoothing.utils.datasets import get_dataset, DATASETS, get_num_classes
from ddsmoothing.utils.models import load_model
from ddsmoothing.certificate import L1Certificate, L2Certificate
from ddsmoothing.smooth import Smooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Certify dataset examples')
    parser.add_argument(
        "--dataset", required=True,
        choices=DATASETS, help="which dataset to use"
    )
    parser.add_argument(
        "--model", required=True,
        type=str, help="path to model of the base classifier"
    )
    parser.add_argument(
        "--model-type", required=True,
        choices=["resnet18", "wideresnet40", "resnet50"],
        type=str, help="type of model to load"
    )
    parser.add_argument(
        "--outfile", required=True,
        type=str, help="output csv file"
    )
    parser.add_argument(
        "--norm", required=True,
        choices=["l1", "l2"], type=str,
        help="norm of the desired certificate"
    )
    parser.add_argument(
        "--method", required=True,
        choices=["fixed", "isotropic_dd"], type=str,
        help="method to obtain the certificate"
    )
    parser.add_argument(
        "--sigma", type=float, required=True,
        help="noise hyperparameter, required for initialization " +
        "in isotropic_dd and ancer"
    )
    parser.add_argument(
        "--optimized-sigmas", type=str,
        help="if certifying with isotropic_dd, pass the optimized " +
        "sigmas directly"
    )

    # dataset options
    parser.add_argument(
        "--dataset-folder-path", type=str, default=None,
        help="dataset folder path, required for ImageNet"
    )
    parser.add_argument(
        "--skip", type=int, default=1,
        help="skip examples in the dataset"
    )
    parser.add_argument(
        "--max", type=int, default=-1,
        help="stop after a certain number of examples"
    )
    parser.add_argument(
        "--split", choices=["train", "test"],
        default="test", help="train or test set"
    )

    # certification parameters
    parser.add_argument(
        "--batch-sz", type=int,
        default=1000, help="certification batch size"
    )
    parser.add_argument(
        "--N0", type=int, default=100
    )
    parser.add_argument(
        "--N", type=int, default=100000,
        help="number of samples to use"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.001,
        help="failure probability"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the base classifier
    num_classes = get_num_classes(args.dataset)
    model = load_model(args.model, args.model_type, num_classes, device=device)

    # get the dataset
    dataset = get_dataset(
        args.dataset,
        args.split,
        folder=args.dataset_folder_path
    )

    # get the type of certificate
    certificate = L1Certificate(device=device) if args.norm == "l1" else \
        L2Certificate(1, device=device)

    if args.norm == "l1":
        args.sigma *= (3**-0.5)

    # prepare output file
    f = open(args.outfile, 'w')
    print(
        "idx\tlabel\tpredict\tradius\tcorrect\ttime",
        file=f,
        flush=True
    )

    if args.method == "fixed":
        sigma = torch.Tensor([args.sigma]).to(device)
    elif args.method == "isotropic_dd":
        if args.optimized_sigmas is None:
            raise ValueError(
                "please provide the path to the isotropic_dd sigmas " +
                "(single file)"
            )

        all_sigmas = torch.load(args.optimized_sigmas)

    for i in tqdm(range(len(dataset))):
        # only certify every args.skip examples, and stop after args.max
        # examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()

        # load sigma for point i in data-dependent methods
        if args.method == "isotropic_dd":
            sigma = torch.Tensor([all_sigmas[i].item()]).to(device)

        before_time = time()

        # certify the point
        smoothed_classifier = Smooth(model, num_classes, sigma, certificate)
        prediction, gap = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch_sz
        )
        after_time = time()

        # compute radius
        correct = int(prediction == label)
        radius = sigma.item() * gap

        time_elapsed = str(datetime.timedelta(
            seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i,
                label,
                prediction,
                radius,
                correct,
                time_elapsed),
            file=f,
            flush=True
        )

    f.close()
