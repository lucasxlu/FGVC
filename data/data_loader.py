import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('../')
from config.cfg import cfg
from data.datasets import PlantsDataset


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
                                      transform=transforms.Compose([
                                          transforms.Resize(227),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ColorJitter(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                                              std=[1, 1, 1])
                                      ]))
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = PlantsDataset(type='validation',
                                    transform=transforms.Compose([
                                        transforms.Resize(227),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ColorJitter(),
                                        transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                                            std=[1, 1, 1])
                                    ]))
        testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        print('Invalid Dataset Name!!!')
        sys.exit(0)

    return trainloader, testloader
