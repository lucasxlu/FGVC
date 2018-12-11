import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

sys.path.append('../')
from fmnist.cfg import cfg

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(227),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def load_fashion_mnist_dataset():
    """
    load plants disease dataset
    :return:
    """
    batch_size = cfg['batch_size']

    train_dataset = FashionMNIST(train=True, transform=data_transforms['train'], root=cfg['image_dir'], download=True)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = FashionMNIST(train=False, transform=data_transforms['test'], root=cfg['image_dir'], download=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader
