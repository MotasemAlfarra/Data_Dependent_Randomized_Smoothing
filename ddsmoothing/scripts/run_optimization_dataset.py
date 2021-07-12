import argparse
import torch
from ddsmoothing.utils.datasets import DATASETS, get_num_classes, cifar10, \
    ImageNet
from ddsmoothing.utils.models import load_model
from ddsmoothing.certificate import L1Certificate, L2Certificate
from ddsmoothing.optimize_dataset import OptimizeIsotropicSmoothingParameters


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
        "--norm", required=True,
        choices=["l1", "l2"], type=str,
        help="norm of the desired certificate"
    )
    parser.add_argument(
        "--isotropic-file", required=True,
        type=str, help="isotropic_dd file for the optimized thetas"
    )
    parser.add_argument(
        "--initial-theta", required=True,
        type=float, help="initial theta value"
    )

    # dataset options
    parser.add_argument(
        "--folder-path", type=str, default=None,
        help="dataset folder path, required for ImageNet"
    )

    # optimization options
    parser.add_argument(
        "--iterations", type=int,
        default=900, help="optimization iterations per sample"
    )
    parser.add_argument(
        "--batch-sz", type=int,
        default=128, help="optimization batch size"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float,
        default=0.04, help="optimization learning rate"
    )
    parser.add_argument(
        "-n", "--num-samples", type=float,
        default=100, help="number of samples per example and iteration"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the base classifier
    num_classes = get_num_classes(args.dataset)
    model = load_model(args.model, args.model_type, num_classes, device=device)
    model = model.eval()

    # get the dataset
    if args.dataset == "cifar10":
        _, test_loader, img_sz, _, testset_len = cifar10(
            args.batch_sz
        )
    else:
        _, test_loader, img_sz, _, testset_len = ImageNet(
            args.batch_sz,
            directory=args.folder_path
        )

    # get the type of certificate
    certificate = L1Certificate(device=device) if args.norm == "l1" else \
        L2Certificate(1, device=device)

    # use initial theta for every point in the dataset
    initial_thetas = args.initial_theta * torch.ones(testset_len).to(device)

    # perform the isotropic optimization
    isotropic_obj = OptimizeIsotropicSmoothingParameters(
        model, test_loader, device=device
    )
    isotropic_obj.run_optimization(
        certificate, args.learning_rate, initial_thetas,
        args.iterations, args.num_samples, args.isotropic_file
    )
