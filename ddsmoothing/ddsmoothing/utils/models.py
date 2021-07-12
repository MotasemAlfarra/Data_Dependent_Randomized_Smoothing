import torch
from torchvision.models.resnet import resnet50

from .resnet import resnet18
from .wideresnet import WideResNet


def get_model(
        model_type: str, num_classes: int, device: str = "cuda"
) -> torch.nn.Module:
    """Returns an instance of a model given the type

    Args:
        model_type (str): model identifier
        num_classes (int): number of classes
        device (str, optional): device where the model will be stored
    """
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes).to(device)
    elif model_type == "wideresnet40":
        model = WideResNet(
            depth=40,
            widen_factor=2,
            num_classes=num_classes
        ).to(device)
    elif model_type == "resnet50":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
    else:
        raise ValueError("model_type requested not available")

    return model


def load_model(
        path: str, model_type: str, num_classes: int, device: str = "cuda"
) -> torch.nn.Module:
    """Obtain the model instance and load the pre-trained parameters

    Args:
        path (str): path to the checkpoint containing ``state_dict``
        model_type (str): model identifier
        num_classes (int): number of classes
        device (str, optional): device where the model will be stored
    """
    # crate model base on model type
    model = get_model(model_type, num_classes, device=device)

    # load the model itself
    if model_type == "resnet50":
        checkpoint = torch.load(path, map_location=device)

        keys = list(checkpoint['state_dict'].keys())
        count = 0
        for key in model.state_dict().keys():
            model.state_dict()[key].copy_(
                checkpoint['state_dict'][keys[count]].data
            )
            count += 1
    elif model_type in ["resnet18", "wideresnet40"]:
        # load the checkpoint
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    return model
