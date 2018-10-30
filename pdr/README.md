# Plants Disease Recognition
## Introduction
This module holds the code for
[Plants Disease Recognition](https://challenger.ai/competition/pdr2018) in AI
 Challenger Competition 2019.


## Solutions
1. Directly train a ```60 category``` deep classifier powered by 
```ResNet/ResNeXt```.
2. Follow a ```coarse-to-fine``` manner. Namely, we split the original 
categories into several coarse categories by ```plants species``` first. And 
then split by ```disease type```, and finally by ```disease extent```.  

### Coarse-to-fine
* 苹果
    * 健康
    * 黑星病
        * 一般
        * 严重
    * 灰斑病
    * 雪松锈病
        * 一般
        * 严重
* 樱桃
    * 健康
    * 白粉病
        * 一般
        * 严重 
* 玉米
    * 健康
    * 灰斑病
        * 一般
        * 严重
    * 锈病    
        * 一般
        * 严重
    * 叶斑病
        * 一般
        * 严重
    * 花叶病毒病
* 葡萄
    * 健康
    * 黑腐病
        * 一般
        * 严重
    * 轮斑病
        * 一般
        * 严重
    * 褐斑病
        * 一般
        * 严重
* 柑桔
    * 健康
    * 黄龙病
        * 一般
        * 严重
* 桃
    * 健康
    * 疮痂病
        * 一般
        * 严重
* 辣椒
    * 健康
    * 疮痂病
        * 一般
        * 严重
* 马铃薯
    * 健康
    * 早疫病
        * 一般
        * 严重
    * 晚疫病
        * 一般
        * 严重
* 草莓
    * 健康
    * 叶枯病
        * 一般
        * 严重
* 番茄
    * 健康
    * 白粉病
        * 一般        
        * 严重
    * 疮痂病
        * 一般        
        * 严重
    * 早疫病
        * 一般        
        * 严重
    * 晚疫病菌        
        * 一般        
        * 严重
    * 叶霉病
        * 一般        
        * 严重
    * 斑点病
        * 一般        
        * 严重
    * 斑枯病
        * 一般        
        * 严重
    * 红蜘蛛损伤
        * 一般        
        * 严重
    * 黄化曲叶病毒病
        * 一般        
        * 严重
    * 花叶病毒病

    
## Performance
| Methods | Model | Stage | Accuracy |
| :---: | :---: | :---: | :---: | 
| 60 Classification | ResNet18 | val | - |
| Coarse-to-fine | ResNet18 | val | - |