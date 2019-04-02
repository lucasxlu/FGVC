# Food Recognition
## Introduction
This module holds the code for
[Food Recognition](https://sites.google.com/view/fgvc5/competitions/fgvcx/ifood) in CVPR'18 Workshop.


## Solutions
1. Directly train a ```61 category``` deep classifier powered by 
```ResNet/ResNeXt```.
2. Follow a ```coarse-to-fine``` manner. Namely, we split the original 
categories into several coarse categories by ```plants species``` first. And 
then split by ```disease type```, and finally by ```disease extent```.  

    
## Performance
| Model | Epoch | Accuracy |
| :---: | :---: | :---: | 
| SEResNet |  |  |
| DenseNet + A-Softmax |  |  |