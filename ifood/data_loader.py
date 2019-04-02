import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from ifood.cfg import cfg
from ifood.datasets import iFoodDataset, iFoodInferenceDataset

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(227),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(1, 1, 1),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(224),
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


def load_ifood_dataset():
    """
    load iFood dataset
    :return:
    """
    batch_size = cfg['batch_size']

    train_dataset = iFoodDataset(train_val='train',
                                 transform=data_transforms['train'])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = iFoodDataset(train_val='val',
                               transform=data_transforms['val'])
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return {"train": trainloader, "val": valloader, "test": valloader}


def load_ifood_inference():
    """
    load iFood inference dataset
    :return:
    """
    batch_size = cfg['batch_size']

    inference_dataset = iFoodInferenceDataset(transform=data_transforms['test'])
    inferenceloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return inferenceloader
