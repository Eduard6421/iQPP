#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------- Imports --------------------------------------------------------------
import os
import pandas as pd
import torch
import skimage.io as sk
import scipy.stats as stats
import numpy as np
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import pickle
import random
import Models.MAE
import argparse

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Resize,Compose, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from math import floor
from PIL import Image


# # Masked Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--model', required=True,help='Dataset on which you want to train the model',choices=['masked','denoising'])
parser.add_argument('--mode', required=True,help='train/run', choices=['train','run'])
args = vars(parser.parse_args())


RESHAPE_SIZE = 512
# SfM + GeM model
DATASET_FOLDERNAME = None
if(args['dataset'] == 'roxford5k'):
    DATASET_FOLDERNAME = 'roxford5k'
if(args['dataset'] == 'rparis6k'):
    DATASET_FOLDERNAME = 'rparis6k'
if(args['dataset'] == 'pascalvoc_700_medium'):
    DATASET_FOLDERNAME = 'pascalvoc'
if(args['dataset'] == 'caltech101_700'):
    DATASET_FOLDERNAME = 'caltech101'

# SfM + GeM model
GND_PATH = "../../Datasets/{}/gnd_{}.pkl".format(DATASET_FOLDERNAME,args['dataset'])
DATASET_FOLDERPATH = '../../Datasets/{}/jpg/'.format(DATASET_FOLDERNAME)
SAVE_PATH = 'Pretrained_Models/{}.torch'.format(args['dataset'])


# In[4]:


# -------------------------------------------------------- Global settings --------------------------------------------------------------
warnings.simplefilter("always")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())


# In[5]:


with open(GND_PATH, 'rb') as handle:
    gnd_file = pickle.load(handle)


# In[6]:


# ------------------------------------------------------------ Model Load  --------------------------------------------------------------
model = torch.load(SAVE_PATH)
model.eval()
criterion = nn.MSELoss()


# In[7]:


# -------------------------------------------------------- Custom Transforms --------------------------------------------------------------

class ExpandDimension(object):
    def __call__(self, sample):
        if(sample.shape[0] == 1):
            sample = sample.repeat(3,1,1)
        return sample    
        
# ---- Dataset Transforms ----
content_transform = Compose([Resize((RESHAPE_SIZE,RESHAPE_SIZE)), ExpandDimension()])    
tensor_transform = ToTensor()


# In[8]:


query_images = gnd_file['qimlist']
details = gnd_file['gnd']

df = pd.DataFrame(columns=['path','score'])

for i in range(len(query_images)):
    print(i)
    image_path = os.path.join(DATASET_FOLDERPATH,query_images[i]) + '.jpg'

    image = Image.open(image_path)
    image = tensor_transform(image)
    
    
    if(gnd_file['gnd'][i]['bbx'] is None or None in gnd_file['gnd'][i]['bbx']):
        crop = image
    else:
        bbox = gnd_file['gnd'][i]['bbx']
        [xmin,ymin,xmax,ymax] = bbox
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        crop = image[:,ymin:ymax,xmin:xmax]
    
    crop = content_transform(crop)
    crop = crop.unsqueeze(0)
    crop = crop.to(device)
    
    loss, pred, mask = model(crop)
    
    new_df = pd.DataFrame([{
        'path' : image_path,
        'score': loss.cpu().detach().numpy()
    }])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)
        #query_image = qimlit


# In[9]:


df.to_csv('../../Results/masked-autoencoder-{}.csv'.format(args['dataset']), index=False)


# In[ ]:





# In[ ]:




