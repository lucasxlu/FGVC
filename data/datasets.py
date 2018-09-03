import os
import sys
import json

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

sys.path.append('../')
from config.cfg import cfg


class PlantsDataset(Dataset):
    """
    Xingse Plants dataset
    https://github.com/visipedia/fgvcx_flower_comp
    Note: testset has no annotations
    """

    def __init__(self, type=True, transform=None):
        train_json = os.path.join(cfg['config']['FGVC']['root'], 'FGVC', 'train.json')
        test_json = os.path.join(cfg['config']['FGVC']['root'], 'FGVC', 'test.json')

        images = []
        labels = []
        with open(train_json, mode='rt', encoding='utf-8') as f:
            train_cfg = json.load(f)
        for i in range(len(train_cfg['images'])):
            images.append(
                os.path.join(cfg['config']['FGVC']['root'], 'FGVC', 'train', train_cfg['images'][i]['file_name'][1:]))
            labels.append(train_cfg['annotations'][i]['category_id'])

        train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.3,
                                                                              random_state=42)

        if type == 'train':
            images = train_images
            labels = train_labels
        elif type == 'validation':
            images = val_images
            labels = val_labels
        else:
            with open(test_json, mode='rt', encoding='utf-8') as f:
                test_cfg = json.load(f)
            for i in range(len(test_cfg['images'])):
                images.append(
                    os.path.join(cfg['config']['FGVC']['root'], 'FGVC', 'test', test_cfg['images'][i]['file_name']))

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(self.images[idx])
        sample = {'image': image, "label": self.labels[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
