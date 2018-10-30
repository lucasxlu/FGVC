"""
definition of datasets
Author: XuLu
"""
import os
import sys
import json

import numpy as np
from PIL import Image
from PIL import ImageFile
from skimage import io
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import shutil

sys.path.append('../')
from pdr.cfg import cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PlantsDiseaseDataset(Dataset):
    """
    Plants Disease Dataset
    """

    def __init__(self, train_val='train', transform=None):
        """
        PyTorch Dataset definition
        :param train_val:
        :param transform:
        """
        train_json = os.path.join(cfg['image_dir'], 'ai_challenger_pdr2018_train_annotations_20181021.json')
        val_json = os.path.join(cfg['image_dir'], 'ai_challenger_pdr2018_validation_annotations_20181021.json')

        imgs = []
        lbs = []

        if train_val == 'train':
            with open(train_json, mode='rt') as f:
                for _ in json.load(f):
                    imgs.append(_['image_id'])
                    lbs.append(_['disease_class'])

            self.img_files = imgs
            self.labels = lbs
        elif train_val == 'val':
            with open(val_json, mode='rt') as f:
                for _ in json.load(f):
                    imgs.append(_['image_id'])
                    lbs.append(_['disease_class'])

            self.img_files = imgs
            self.labels = lbs
        else:
            print('Invalid data type. Since it only supports [train/val]...')
            sys.exit(0)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['image_dir'], 'AgriculturalDisease_testA', 'images', self.img_files[idx]))
        label = self.labels[idx]

        sample = {'image': image, 'label': label, 'filename': self.img_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class PlantsDiseaseInferenceDataset(Dataset):
    """
    UNFINISHED YET!!
    Plants Disease Inference dataset
    """

    def __init__(self, transform=None):
        """
        PyTorch Dataset definition
        :param transform:
        """
        inference_base = '/var/log/PDR'
        dir_names = []
        img_files = []
        labels = []
        for i, dir_name in enumerate(dir_names):
            for _ in os.listdir(os.path.join(inference_base, dir_name)):
                img_files.append(os.path.join(inference_base, dir_name, _))
                labels.append(i)

        self.img_files = img_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        print(self.img_files[idx])

        image = io.imread(self.img_files[idx])
        label = self.labels[idx]
        sample = {'image': image, 'label': label, 'class': round(label) - 1, 'filename': self.img_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
