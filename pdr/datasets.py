"""
definition of datasets
Author: LucasX
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from pdr.cfg import cfg


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
            with open(train_json, mode='rt', encoding='utf-8') as f:
                for _ in json.load(f):
                    img_fp = os.path.join(cfg['image_dir'], 'AgriculturalDisease_trainingset', 'images',
                                          _['image_id']).encode('ascii', 'ignore').decode('utf-8')
                    if os.path.exists(img_fp):
                        imgs.append(img_fp)
                        lbs.append(_['disease_class'])

            self.img_files = imgs
            self.labels = lbs
        elif train_val == 'val':
            with open(val_json, mode='rt', encoding='utf-8') as f:
                for _ in json.load(f):
                    img_fp = os.path.join(cfg['image_dir'], 'AgriculturalDisease_validationset', 'images',
                                          _['image_id']).encode('ascii', 'ignore').decode('utf-8')
                    if os.path.exists(img_fp):
                        imgs.append(img_fp)
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
        img_path = self.img_files[idx]

        image = io.imread(img_path)
        label = self.labels[idx]

        sample = {'image': image, 'label': label, 'filename': self.img_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class PlantsDiseaseInferenceDataset(Dataset):
    """
    Plants Disease Inference dataset
    """

    def __init__(self, transform=None):
        """
        PyTorch Dataset definition
        :param transform:
        """
        inference_base = '/var/log/PDR'
        img_files = []
        for img_f in os.listdir(os.path.join(inference_base, 'AgriculturalDisease_testA', 'images')):
            # img_fp = os.path.join(inference_base, 'AgriculturalDisease_testA', 'images', img_f)
            img_fp = os.path.join(inference_base, 'AgriculturalDisease_testA', 'images', img_f).encode('ascii',
                                                                                                       'ignore').decode(
                'utf-8')
            if os.path.exists(img_fp):
                img_files.append(img_fp)

        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = io.imread(self.img_files[idx])
        sample = {'image': image, 'filename': self.img_files[idx]}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
