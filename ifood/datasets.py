"""
definition of iFood dataset
Author: LucasX
"""
import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from ifood.cfg import cfg


class iFoodDataset(Dataset):
    """
    iFood Dataset
    """

    def __init__(self, train_val='train', transform=None):
        """
        PyTorch Dataset definition
        :param train_val:
        :param transform:
        """
        train_csv = os.path.join(cfg['image_dir'], 'annotation', 'train_info.csv')
        val_csv = os.path.join(cfg['image_dir'], 'annotation', 'val_info.csv')

        imgs = []
        lbs = []

        if train_val == 'train':
            with open(train_csv, mode='rt', encoding='utf-8') as f:
                for _ in f.readlines():
                    imgs.append(os.path.join(cfg['image_dir'], 'train_set', _.split(",")[0]))
                    lbs.append(int(_.split(",")[1].strip()))

            self.img_files = imgs
            self.labels = lbs
        elif train_val == 'val':
            with open(val_csv, mode='rt', encoding='utf-8') as f:
                for _ in f.readlines():
                    imgs.append(os.path.join(cfg['image_dir'], 'val_set', _.split(",")[0]))
                    lbs.append(int(_.split(",")[1].strip()))

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


class iFoodInferenceDataset(Dataset):
    """
    iFood Inference dataset
    """

    def __init__(self, transform=None):
        """
        PyTorch Dataset definition
        :param transform:
        """
        inference_base = os.path.join(cfg['image_dir'], 'test_set')
        img_files = []
        for img_f in os.listdir(inference_base):
            img_fp = os.path.join(inference_base, img_f)
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
