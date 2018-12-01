from pprint import pprint

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def infer(img_file):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 998)

    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./model/ResNet18_Plant.pth'))

    model.eval()
    model.to(device)

    df = pd.read_csv('../label.csv')
    key_type = {}
    for i in range(len(df['category_name'].tolist())):
        key_type[int(df['category_name'].tolist()[i].split('_')[-1])] = df['label'].tolist()[i]

    img = Image.open(img_file)

    preprocess = transforms.Compose([
        transforms.Resize(227),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = preprocess(img)
    img.unsqueeze_(0)

    img = img.to(device)

    outputs = model(img)
    outputs = F.softmax(outputs, dim=1)

    # get TOP-K output labels and corresponding probabilities
    topK_prob, topK_label = torch.topk(outputs, 5)
    prob = topK_prob.to("cpu").detach().numpy().tolist()

    _, predicted = torch.max(outputs.data, 1)

    return {
        'status': 0,
        'message': 'success',
        'results': [
            {
                'name': key_type[int(topK_label[0][i].to("cpu")) + 1],
                'category id': int(topK_label[0][i].data.to("cpu").numpy()),
                'prob': round(prob[0][i], 4)
            } for i in range(5)
        ]
    }


if __name__ == '__main__':
    pprint(infer('./14.jpg'))

    # idx = 1
    # import os
    # import shutil
    #
    # dir_name = 'G:\Dataset\CV\FGCV5/train\category_992'
    #
    # for _ in os.listdir(dir_name):
    #     shutil.move(os.path.join(dir_name, _), os.path.join(dir_name, str(idx) + '.' + _.split('.')[-1]))
    #     idx += 1
