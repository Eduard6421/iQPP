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
import argparse


from DataLoaders.Loaders import train_dataloader,test_dataloader,validation_dataloader
from Models.DAE import DenoisingAutoencoder
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Resize,Compose, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from PIL import Image



# -------------------------------------------------------- Global settings --------------------------------------------------------------
warnings.simplefilter("always")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())


# Denoising Autoencoder Result


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--model', required=True,help='Dataset on which you want to train the model',choices=['masked','denoising'])
parser.add_argument('--mode', required=True,help='train/run', choices=['train','run'])
args = vars(parser.parse_args())

NUM_CHANNELS = 64
RESHAPE_SIZE = 512


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


with open(GND_PATH, 'rb') as handle:
    gnd_file = pickle.load(handle)


# ------------------------------------------------------------ Model Load  --------------------------------------------------------------
model = DenoisingAutoencoder(num_channels = NUM_CHANNELS)
model.to(device)
model = torch.load(SAVE_PATH) #model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
criterion = nn.MSELoss()


# -------------------------------------------------------- Custom Transforms --------------------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=.35):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noised_tensor = tensor + (torch.randn(tensor.size()) * self.std + self.mean).to(device)
        return torch.clip(noised_tensor, min= 0.0, max=1.0)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ExpandDimension(object):
    def __call__(self, sample):
        if(sample.shape[0] == 1):
            sample = sample.repeat(3,1,1)
        return sample    
    
# -------------------------------------------------------- Custom Operations  ------------------------------------------------------------
def add_noise(image, bitmask_size):
    shape = image.shape
    height = shape[1]
    width = shape[2]
    bitmask = (torch.FloatTensor(height,width).uniform_() > bitmask_size).to(device)
    return torch.mul(image,bitmask)    
    
# ---- Dataset Transforms ----
content_transform = Compose([Resize((RESHAPE_SIZE,RESHAPE_SIZE)), ExpandDimension() ])
tensor_transform = ToTensor()
noise = AddGaussianNoise()


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
    crop = crop.to(device)    
    noised_crop = noise(crop)
    noised_crop = noised_crop.unsqueeze(0)

    #plt.figure()
    #plt.imshow(image.permute(1, 2, 0))
    #plt.figure()
    #plt.imshow(crop.cpu().squeeze(0).permute(1, 2, 0))
    #print(i)
    #if(i==2):
    #    raise Exception('end')
    
    output = model(noised_crop)
    loss = criterion(crop.unsqueeze(0),output)
    
    
    new_df = pd.DataFrame([{
        'path' : image_path,
        'score': loss.cpu().detach().numpy()
    }])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)
        #query_image = qimlit



df.to_csv('../../Results/denoising-autoencoder-{}.csv'.format(args['dataset']), index=False)