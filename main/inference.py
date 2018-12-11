# Model Inference
import time
import base64
from pprint import pprint

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from torchvision import models


class PlantRecognizer():
    """
    Plant Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path):
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 998)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.load_state_dict(torch.load(pretrained_model_path))

        # if torch.cuda.device_count() > 1:
        #     print("We are running on", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        #     model.load_state_dict(torch.load(pretrained_model_path))
        # else:
        #     state_dict = torch.load(pretrained_model_path)
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #
        #     model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        df = pd.read_csv('../label.csv')
        key_type = {}
        for i in range(len(df['category_name'].tolist())):
            key_type[int(df['category_name'].tolist()[i].split('_')[-1]) - 1] = df['label'].tolist()[i]

        self.device = device
        self.model = model
        self.key_type = key_type
        self.topK = 5

    def infer(self, img_file):
        tik = time.time()
        img = io.imread(img_file)
        img = Image.fromarray(img.astype(np.uint8))

        preprocess = transforms.Compose([
            transforms.Resize(227),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        outputs = self.model.forward(img)
        outputs = F.softmax(outputs, dim=1)

        # get TOP-K output labels and corresponding probabilities
        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        tok = time.time()

        return {
            'status': 0,
            'message': 'success',
            'elapse': tok - tik,
            'results': [
                {
                    'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                    'category id': int(topK_label[0][i].data.to("cpu").numpy()) + 1,
                    'prob': round(prob[0][i], 4)
                } for i in range(self.topK)
            ]
        }

    def infer_from_img_url(self, img_url):
        tik = time.time()
        response = requests.get(img_url, timeout=20)
        if response.status_code in [403, 404, 500]:
            return {
                'status': 2,
                'message': 'Invalid URL',
                'elapse': time.time() - tik,
                'results': None
            }

        else:
            img_content = response.content

            import io
            from PIL import Image
            img_np = np.array(Image.open(io.BytesIO(img_content)))
            img = Image.fromarray(img_np.astype(np.uint8))

            preprocess = transforms.Compose([
                transforms.Resize(227),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img = preprocess(img)
            img.unsqueeze_(0)

            img = img.to(self.device)

            outputs = self.model.forward(img)
            outputs = F.softmax(outputs, dim=1)

            # get TOP-K output labels and corresponding probabilities
            topK_prob, topK_label = torch.topk(outputs, self.topK)
            prob = topK_prob.to("cpu").detach().numpy().tolist()

            _, predicted = torch.max(outputs.data, 1)

            tok = time.time()

            return {
                'status': 0,
                'message': 'success',
                'elapse': tok - tik,
                'results': [
                    {
                        'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                        'category id': int(topK_label[0][i].data.to("cpu").numpy()) + 1,
                        'prob': round(prob[0][i], 4)
                    } for i in range(self.topK)
                ]
            }


if __name__ == '__main__':
    plant_recognizer = PlantRecognizer('./model/ResNet50_Plant.pth')
    # pprint(plant_recognizer.infer('./{0}.jpg'.format(5)))
    pprint(plant_recognizer.infer_from_img_url(
        "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1544967248&di=f378a35f9c1749a96eb8ca5c4a17c4e9&imgtype=jpg&er=1&src=http%3A%2F%2Fimgsrc.baidu.com%2Fimgad%2Fpic%2Fitem%2F8718367adab44aed9d3ad930b91c8701a08bfbea.jpg"))

    # for i in range(4):
    #     pprint(plant_recognizer.infer('./{0}.jpg'.format(i + 1)))
    #     print('=' * 100)
