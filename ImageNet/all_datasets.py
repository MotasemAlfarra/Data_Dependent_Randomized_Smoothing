from torchvision import transforms
from torchvision import datasets 
from torch.utils.data import DataLoader, Dataset
import os

def ImageNet(path, batch_sz):
    img_sz = [3, 224, 224]
    trainset, testset = ImageNet_Trainset(path), ImageNet_Testset(path)
    print('length of trainset and test set is {}, {}'.format(len(trainset), len(testset)))
    train_loader = DataLoader(trainset,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=2)
    return train_loader, test_loader, img_sz, len(trainset), len(testset)


class ImageNet_Trainset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.imgnet = datasets.ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self):
        return len(self.imgnet)

class ImageNet_Testset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "val")
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        self.imgnet = datasets.ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target, index

    def __len__(self):
        return len(self.imgnet)
