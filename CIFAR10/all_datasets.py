from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


# CIFAR10
class CIF10_TRAINSET(Dataset):
    def __init__(self):
        t_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
        self.cifar10 = CIFAR10(root='datasets',
                               download=True,
                               train=True,
                               transform=t_train)

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIF10_TESTSET(Dataset):
    def __init__(self):
        self.cifar10 = CIFAR10(root='datasets',
                               download=True,
                               train=False,
                               transform=transforms.ToTensor())

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


def cifar10(batch_sz):
    img_sz = [3, 32, 32]
    trainset, testset = CIF10_TRAINSET(), CIF10_TESTSET()
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=True,
                             pin_memory=True, num_workers=2)
    return train_loader, test_loader, img_sz
