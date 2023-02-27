#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------- Imports --------------------------------------------------------------
import os
import pandas as pd
import skimage.io as sk
import scipy.stats as stats
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import warnings
import random
import argparse

from DataLoaders.Loaders import train_dataloader,test_dataloader,validation_dataloader
from Models.DAE import DenoisingAutoencoder
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Resize,Compose, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from math import floor
from random import randrange

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--model', required=True,help='Dataset on which you want to train the model',choices=['masked','denoising'])
parser.add_argument('--mode', required=True,help='train/run', choices=['train','run'])
args = vars(parser.parse_args())

# -------------------------------------------------------- Global settings --------------------------------------------------------------
warnings.simplefilter("always")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())

SAVE_PATH = 'Pretrained_Models/{}.torch'.format(args['dataset'])
NUM_CHANNELS = 64
NUM_EPOCHS = 15


# In[3]:



# In[4]:


# -------------------------------------------------------- Custom Operations  ------------------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=.35):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noised_tensor = tensor + (torch.randn(tensor.size()) * self.std + self.mean).to(device)
        return torch.clip(noised_tensor, min= 0.0, max=1.0)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# In[5]:


# ------------------------------------------------------------ Model Instantiation  --------------------------------------------------------------
model = DenoisingAutoencoder(num_channels = NUM_CHANNELS)
model.to(device)


# In[6]:


# ------------------------------------------------------------ Loss / Optimizer Instantiation  --------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[7]:


torch.cuda.empty_cache()
noise_transform = AddGaussianNoise()


# In[8]:

# In[9]:


# ------------------------------------------------------------ Training  --------------------------------------------------------------

lowest_validation_score = 100000

for idx,epoch in enumerate(range(NUM_EPOCHS)):
    
    model.train()
    
    for batch_id,image in enumerate(train_dataloader):
        
        image = image.to(device)
        xs  = noise_transform(image)

        recon = model(xs)
        loss = criterion(recon, image)
        
        xs.detach()
        image.detach()
        recon.detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(batch_id%50 == 0):
            print(f"Training batch {batch_id+1}/{len(train_dataloader)}".format())
            print(f"Training loss {loss.item():.5f}".format())        
            
    if(epoch%2 == 0):
        
        num = randrange(image.shape[0])
        
        imgplot = plt.imshow(image[num,:,:,:].cpu().permute(1,2,0))
        plt.title('Original Image')
        plt.show()
            
        imgplot = plt.imshow(xs[num,:,:,:].cpu().permute(1,2,0))
        plt.title('Masked Image')
        plt.show()            
        
        imgplot = plt.imshow(recon[num,:,:,:].cpu().detach().permute(1,2,0))
        plt.title('Reconstituted Image')
        plt.show()        
            
             
    model.eval()
        
    total_loss = 0
    for image in validation_dataloader:
        
        image = image.to(device)
        xs  = noise_transform(image)
        
        recon = model(xs)
        loss = criterion(recon,image)

        xs.detach()
        image.detach()
        recon.detach()
        
        total_loss += loss.item()
    
    validation_loss = total_loss / len(validation_dataloader)
    
    if(validation_loss < lowest_validation_score):
        torch.save(model,SAVE_PATH)
        lowest_validation_score = validation_loss 
     
    print(f'Epoch:{epoch+1}, Validation Loss:{loss.item():.4f}')