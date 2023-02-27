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
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Resize,Compose, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from torchvision.models import vgg16,VGG16_Weights
from joblib import load

import matplotlib.pyplot as plt
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
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

class VGGExtractor():
    
    def __init__(self, ):
    
        # Pytorch exposes easy preprocess steps so that we can match the input data
        self.vgg16_model = vgg16(weights=VGG16_Weights.DEFAULT)

        # Remove last linear and keep only 4096d vectors
        self.vgg16_model.classifier = self.vgg16_model.classifier[:-1]
        self.vgg16_model.to(device)
    
        for parameter in self.vgg16_model.parameters():
            parameter.requires_grad = False
        
        self.vgg16_model.eval()
        
    def __call__(self, image):
        
        return self.vgg16_model(image)


# In[4]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


extractor = VGGExtractor()
scaler = load('scaler.joblib')
svr  = load('svr.joblib')


# In[5]:


GND_PATH = "../../Datasets/{}/gnd_{}.pkl".format(DATASET_FOLDER,args['dataset'])
DATASET_FOLDERPATH = '../../Datasets/data/test/{}/jpg'.format(DATASET_FOLDER)


# In[6]:


with open(GND_PATH, 'rb') as handle:
    gnd_file = pickle.load(handle)
    
query_images = gnd_file['qimlist']
details = gnd_file['gnd']


# In[7]:


df = pd.DataFrame(columns=['path','score'])

transform_fct = VGG16_Weights.DEFAULT.transforms() 
to_tensor = ToTensor()


for i in range(len(query_images)):
    image_path = os.path.join(DATASET_FOLDERPATH,query_images[i]) + '.jpg'
    image = Image.open(image_path)
    image = to_tensor(image)
    image = image.to(device)
    
    if(image.shape[0] == 1):
        image = torch.cat([image,image,image],axis = 0)
        
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
        
    #print(details[0])
    #plt.imshow(crop.cpu().permute(1,2,0))
    #raise Exception('asd')

    crop = transform_fct(crop)
    
    crop = crop.unsqueeze(0)
    
    model_output = model(crop)
    
    
    scaled_features = scaler.transform(model_output.cpu().numpy())
    
    score = svr.predict(scaled_features)[0]
    
    new_df = pd.DataFrame([{
        'path' : image_path,
        'score': score
    }])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)
        #query_image = qimlit


# In[8]:


df.to_csv('../../Results/ionescu-et-al-{}.csv'.format(args['dataset']), index=False)


# In[ ]:





# In[ ]:





# In[ ]:




