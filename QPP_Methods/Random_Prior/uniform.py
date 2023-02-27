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

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Resize,Compose, ToTensor
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

EMBEDDINGS_FOLDER = None

if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'

DATASET_FOLDER = None

if args['dataset'] == 'roxford5k':
   DATASET_FOLDER = 'roxford5k'
elif args['dataset'] == 'rparis6k':
    DATASET_FOLDER = 'rparis6k'
elif args['dataset'] == 'pascalvoc_700_medium':
    DATASET_FOLDER = 'pascalvoc'
elif args['dataset'] == 'caltech101_700':
    DATASET_FOLDER = 'caltech101'
else:
    raise Exception('Unknown dataset selected')

# In[2]:


# SfM + GeM model
GND_PATH = "/../../Datasets/{}/gnd_{}.pkl".format(DATASET_FOLDER,args['dataset'])
DATASET_FOLDERPATH = '/../../Datasets/{}/jpg/'.format(DATASET_FOLDER)


# In[3]:


with open(GND_PATH, 'rb') as handle:
    gnd_file = pickle.load(handle)
    
query_images = gnd_file['qimlist']
details = gnd_file['gnd']

df = pd.DataFrame(columns=['path','score'])

for i in range(len(query_images)):
    image_path = os.path.join(DATASET_FOLDERPATH,query_images[i]) + '.jpg'
    image = read_image(image_path)
    
    new_df = pd.DataFrame([{
        'path' : image_path,
        'score': np.random.uniform(0,1,1)[0]
    }])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)
        #query_image = qimlit
df.to_csv('/../../Results/random-uniform-results-{}.csv'.format(args['dataset']), index=False)