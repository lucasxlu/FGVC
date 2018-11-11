# Fine-Grained Visual Classification

## Introduction
1. Train [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) to recognize over 998 categories (997 plants + 1 others) for Kaggle Competition.

2. Deep Learning for [Plants Disease Recognition](./pdr) for AI Challenger 
2018 Competition.


## Solutions
* Directly train state-of-the-art models (ResNeXt/ResNet/DenseNet) pretrained
 on ImageNet and fine-tune for classification.
* Apply _bilinear pooling_ and _zoom data augmentation_ for fine-grained visual
 recognition.

## Results

| Model | Top-1 Accuracy |
| :---: | :---: |
| ResNet50 | 89.0947% |
| ResNet18 | 84.5452% |