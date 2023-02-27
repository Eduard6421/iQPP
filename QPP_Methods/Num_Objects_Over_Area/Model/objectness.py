#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.transforms import Resize
from torch.nn import Sequential
import warnings
import torch
import pandas as pd
import torch
import skimage.io as sk
import scipy.stats as stats
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
parser.add_argument('--metric', required=True,help='Retrieval metric',choices=['ap','p@100'])
args = vars(parser.parse_args())

# In[2]:


# -------------------------------------------------------- Global settings --------------------------------------------------------------
warnings.simplefilter("always")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())


# In[3]:


class ObjectnessDifficultyRegressor():
    
    def __init__(self, ):
        
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        
        self.content_transform = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        self.content_transform = Sequential(Resize((224,224)),self.content_transform)

        self.model = fasterrcnn_resnet50_fpn(weights = self.weights, progress=False, trainable_backbone_layers=0)
        self.model.eval()
        self.model.to(device)
        
    
    def __call__(self, image):

        image = image.to(device)
        resized_image = self.content_transform(image)
        result = self.model([resized_image])
        boxes = result[0]['boxes']
        boxes = boxes.tolist()        
        n = 0
        total_area = 0
    
        for x1,y1,x2,y2 in boxes:
            width  = x2 - x1 
            height = y2 - y1
            area = width * height
            n = n + 1
            total_area += area
        
        if(n == 0):
            return 9999999999999
        else:
            return  n * n / total_area
        
        
        

