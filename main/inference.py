import numpy as np

import torch
import torch.nn as nn
from skimage import io
from skimage.transform import resize
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(resnet_model='./model/resnet18.pth'):
    """
    load pretrained model
    :param resnet_model:
    :return:
    """
    resnet = models.resnet18(pretrained=False)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 998)
    print("loading pre-trained model...")
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        resnet = nn.DataParallel(resnet)
        resnet.load_state_dict(torch.load(resnet_model))
    else:
        state_dict = torch.load(resnet_model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        resnet.load_state_dict(new_state_dict)

    resnet.eval()
    resnet = resnet.to(device)

    return resnet


def predict(model, img_file):
    """
    predict with pretrained ResNet18
    :param model:
    :param img_file:
    :return:
    """

    img = resize(io.imread(img_file), (224, 224), mode='constant')

    img[0] -= 131.45376586914062
    img[1] -= 103.98748016357422
    img[2] -= 91.46234893798828

    img = np.transpose(img, [2, 0, 1])

    img = torch.from_numpy(img).unsqueeze(0).float()
    img = img.to(device)

    pred = model.forward(img)
    _, predicted = torch.max(pred.data, 1)

    print(int(predicted.cpu()))


if __name__ == '__main__':
    resnet = load_model()
    predict(resnet, 'syc.jpg')
