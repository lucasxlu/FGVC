import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from config.cfg import cfg
from data.datasets import PlantsDataset

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_data(dataset_name):
    """
    load dataset
    :param dataset_name:
    :return:
    """
    if dataset_name == 'FGVC':
        batch_size = cfg['config']['FGVC']['batch_size']
        print('loading %s dataset...' % dataset_name)
        train_dataset = PlantsDataset(type='train',
                                      transform=data_transforms['train'])
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = PlantsDataset(type='val',
                                    transform=data_transforms['val'])
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        test_dataset = PlantsDataset(type='test',
                                     transform=data_transforms['test'])
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        print('Invalid Dataset Name!!!')
        sys.exit(0)

    return {'train': trainloader, 'val': valloader, 'test': testloader}
