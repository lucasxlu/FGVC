# Model Inference
from pprint import pprint

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


class PlantRecognizer():
    """
    Plant Recognition Class Wrapper
    """

    def __init__(self, pretrained_model_path):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 998)

        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = nn.DataParallel(model)
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
        #         model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        df = pd.read_csv('../label.csv')
        key_type = {}
        for i in range(len(df['category_name'].tolist())):
            key_type[int(df['category_name'].tolist()[i].split('_')[-1])] = df['label'].tolist()[i]

        self.device = device
        self.model = model
        self.key_type = key_type
        self.topK = 5

    def infer(self, img_file):
        # img = io.imread(img_file)
        # img = Image.fromarray(img.astype(np.uint8))
        img = Image.open(img_file)

        preprocess = transforms.Compose([
            transforms.Resize(227),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img.unsqueeze_(0)

        img = img.to(self.device)

        outputs = self.model(img)
        outputs = F.softmax(outputs, dim=1)

        # get TOP-K output labels and corresponding probabilities
        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        return {
            'status': 0,
            'message': 'success',
            'results': [
                {
                    'name': self.key_type[int(topK_label[0][i].to("cpu"))],
                    'prob': round(prob[0][i], 4)
                } for i in range(self.topK)
            ]
        }


if __name__ == '__main__':
    plant_recognizer = PlantRecognizer('./model/ResNet50_Plant.pth')
    pprint(plant_recognizer.infer('./test.jpg'))
