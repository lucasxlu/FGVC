import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append('../')
from util.file_utils import mkdirs_if_not_exist
from config.cfg import cfg
from data import data_loader


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs,
                inference=False):
    """
    train model
    :param model:
    :param train_dataloader:
    :param test_dataloader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        print('Start training ResNet...')
        model.train()

        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                images, label = data['image'], data['label']

                images = images.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                # gender = gender.float().view(cfg['batch_size'], 1)

                pred = model(images)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training ResNet...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'resnet18.pth'))
        print('ResNet has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/resnet18.pth')))

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, label = data['image'], data['label']
            images = images.to(device)
            label = label.to(device)

            pred = model.forward(images)
            _, predicted = torch.max(pred.data, 1)

            total += pred.size(0)

            correct += (predicted == label).sum().item()

    print('Accuracy of ResNet18: %f' % (correct / total))


def run_resnet(train_dataloader, test_dataloader):
    """
    train and eval on ResNet
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 998)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_model(model=resnet, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=cfg['config']['FGVC']['epoch'],
                inference=False)


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_data("FGVC")
    run_resnet(trainloader, testloader)
