import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append('../')
from data.data_loader import load_data
from util.file_utils import mkdirs_if_not_exist
from config.cfg import cfg

dataloaders = load_data('FGVC')
dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}
batch_size = cfg['config']['FGVC']['batch_size']


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
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'resnet.pth'))
        print('ResNet has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/resnet.pth')))

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

    print('Accuracy of ResNet: %f' % (correct / total))


def train_model_ft(model, dataloaders, criterion, optimizer, scheduler, num_epochs, inference):
    """
    train model with fine-tune on ImageNet
    :param dataloaders:
    :param model:
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
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 100)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs = data['image'].to(device)
                    labels = data['label'].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / (dataset_sizes[phase] * batch_size)
                epoch_acc = running_corrects.double() / (dataset_sizes[phase] * batch_size)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model_path_dir = './model'
                    mkdirs_if_not_exist(model_path_dir)
                    torch.save(model.state_dict(), os.path.join(model_path_dir, '{0}_Epoch_{1}.pth'.format(
                        model.__class__.__name__, epoch)))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/%s.pth' % model.__class__.__name__)))

    model.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    filename_list = []
    probs = []

    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, filenames = data['image'], data['label'], data['filename']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            outputs = F.softmax(outputs)

            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, 2)
            probs += topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred += predicted.to("cpu").detach().numpy().tolist()
            y_true += labels.to("cpu").detach().numpy().tolist()
            filename_list += filenames

    print('Accuracy of ResNet: %f' % (correct / total))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    cm = np.array(cm)

    precisions = []
    recalls = []
    for i in range(len(cm)):
        precisions.append(cm[i][i] / sum(cm[:, i].tolist()))
        recalls.append(cm[i][i] / sum(cm[i, :].tolist()))

    print('Precision List: ')
    print(precisions)
    print('Recall List: ')
    print(recalls)

    print("Precision of {0} on val set = {1}".format(model.__class__.__name__,
                                                     sum(precisions) / len(precisions)))
    print(
        "Recall of {0} on val set = {1}".format(model.__class__.__name__, sum(recalls) / len(recalls)))

    print('Output CSV...')
    col = ['filename', 'gt', 'pred', 'prob']
    df = pd.DataFrame([[filenames[i], y_true[i], y_pred[i], probs[i][0]] for i in range(len(filenames))],
                      columns=col)
    df.to_csv("./output-%s.csv" % model.__class__.__name__, index=False)
    print('CSV has been generated...')


def run_resnet(epoch, inference=False):
    """
    train and eval on ResNet
    :return:
    """
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 998)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_model_ft(model=resnet, dataloaders=load_data("FGVC"), criterion=criterion,
                   optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=epoch, inference=inference)

    # train_model(model=resnet, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
    #             criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
    #             num_epochs=cfg['config']['FGVC']['epoch'],
    #             inference=False)


if __name__ == '__main__':
    run_resnet(epoch=200, inference=False)
