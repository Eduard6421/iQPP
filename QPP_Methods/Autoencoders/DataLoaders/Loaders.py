#!/usr/bin/env python
# coding: utf-8


# -------------------------------------------------------- Imports --------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import torch 
import pickle
import argparse

from PIL import Image
from torchvision.transforms import Resize,Compose, ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset,DataLoader,random_split



# -------------------------------------------------------- Global configs --------------------------------------------------------------
RESHAPE_SIZE = 512
BATCH_SIZE = 32

# -------------------------------------------------------- Custom Dataset --------------------------------------------------------------

class ParisDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        
        gnd_path = os.path.join(image_dir, 'gnd_rparis6k.pkl')
        
    
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if(self.transform):
            image = self.transform(image)
        return image

class OxfordDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        image_paths = os.listdir(image_dir)
        gnd_path = os.path.join(image_dir, 'gnd_roxford5k.pkl')
        
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if(self.transform):
            image = self.transform(image)
        return image   
    
class PascalVOCEasyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        gnd_path = os.path.join(image_dir, 'gnd_pascalvoc_700.pkl')
        
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if(self.transform):
            image = self.transform(image)
        return image  
    
    
class PascalVOCMediumDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        gnd_path = os.path.join(image_dir, 'gnd_pascalvoc_700_medium.pkl')
        
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if(self.transform):
            image = self.transform(image)
        return image    
    
    
class PascalVOCHardDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        gnd_path = os.path.join(image_dir, 'gnd_pascalvoc_700_no_bbx.pkl')
        
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if(self.transform):
            image = self.transform(image)
        return image        
    
    
    
class Caltech101Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        gnd_path = os.path.join(image_dir, 'gnd_caltech101_700.pkl')
        
        handle = open(gnd_path,'rb')
        csv_file = pickle.load(handle)
        
        image_names = csv_file['imlist']

        self.image_paths = [ os.path.join(image_dir,'jpg',name+'.jpg') for name in image_names]
        self.transform = transform
        
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if(self.transform):
            image = self.transform(image)
        return image        
    


# -------------------------------------------------------- Custom Transforms --------------------------------------------------------------

class ExpandDimension(object):
    def __call__(self, sample):
        if(sample.shape[0] == 1):
            sample = sample.repeat(3,1,1)
        return sample



# -------------------------------------------------------- Data Loaders  --------------------------------------------------------------

# ---- Dataset Transforms ----
content_transform = Compose([ToTensor(),Resize((RESHAPE_SIZE,RESHAPE_SIZE)),ExpandDimension()])



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--model', required=True,help='Dataset on which you want to train the model',choices=['masked','denoising'])
parser.add_argument('--mode', required=True,help='train/run', choices=['train','run'])
args = vars(parser.parse_args())

dataset = None

if args['dataset'] == 'roxford5k':
    dataset = OxfordDataset(image_dir='../../Datasets/roxford5k',transform=content_transform)
elif args['dataset'] == 'rparis6k':
    dataset = ParisDataset(image_dir='../../Datasets/rparis6k',transform=content_transform)
elif args['dataset'] == 'pascalvoc_700_medium':
    dataset = PascalVOCMediumDataset(image_dir='../../Datasets/pascalvoc', transform=content_transform)
elif args['dataset'] == 'caltech101_700':
    dataset = Caltech101Dataset(image_dir='../../Datasets/caltech101', transform=content_transform)
else:
    raise Exception('Unknown dataset selected')


# ---- Dataset Reads  ----
# ---- Dataset Split  ----
TRAIN_PERCENT = 0.9
VALIDATION_PERCENT = 0.05
TEST_PERCENT = 0.05

train_size = int(TRAIN_PERCENT*int(len(dataset)))
validation_size = int(VALIDATION_PERCENT*int(len(dataset)))
test_size  = len(dataset) - (train_size + validation_size)

train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths=[train_size,validation_size,test_size],generator=torch.Generator().manual_seed(420))



#  ---- Dataloaders ----
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)