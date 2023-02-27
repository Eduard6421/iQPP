#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

from torchvision.models import vit_b_32
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.models import ViT_B_32_Weights
from torch.utils.data import DataLoader, Dataset,DataLoader,random_split
from torch.optim import Adam
from PIL import Image
from torchvision.transforms import Resize,Compose, ToTensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
parser.add_argument('--metric', required=True,help='Retrieval metric',choices=['ap','p@100'])
args = vars(parser.parse_args())

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

# -------------------------------------------------------- Global settings --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())

BATCH_SIZE = 8
RESHAPE_SIZE = 512
NUM_EPOCHS = 50
GND_FILEPATH = '/../../Results/gnd_{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric'])
FOLD_PATH = '/../../Folds/{}-folds.pkl'.format(args['dataset'])


def compute_loss(model , loader):
    
    total_loss = 0.0
    
    with torch.no_grad():
        for data in loader:
            images, scores, paths = data
            
            images = images.to(device,dtype=torch.float)
            scores = scores.to(device).unsqueeze(1)
            
            outputs = model(images)
            
            loss = criterion(outputs, scores)
            total_loss += loss.item()
    
    total_loss /= len(loader)
    
    return total_loss


def train_model(model, train_dataloader, validation_dataloader, optimizer , criterion):
    
    min_loss = 1000
    for i in range(NUM_EPOCHS):
        
        epoch_train_loss = 0
        
        for idx, data in enumerate(train_dataloader):
            #print("Batch num {}/{}".format(idx+1, len(train_dataloader)))
 
            (images,scores,img_path) = data
    
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(scores, outputs)

            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_dataloader)
        epoch_validation_loss = compute_loss(model, validation_dataloader)
        
        if(i % 1== 0):
            print("Epoch num {}/{}".format(i+1,NUM_EPOCHS))
            print("Epoch train loss {}".format(epoch_train_loss))
            print("Epoch validation loss {}".format(epoch_validation_loss))


class DifficultyFoldDataset(Dataset):

    def __init__(self, data, transform=None):
        self.image_paths = data[:,0]
        self.scores = data[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if(self.transform):
            image = self.transform(image)
        
        score = torch.tensor(float(self.scores[idx]))

        return (image, score, img_path)


content_transform = Compose([ViT_B_32_Weights.IMAGENET1K_V1.transforms()])

train_df = pd.read_csv(GND_FILEPATH)
dataset = np.array(train_df[['path','score']].values.tolist())
to_tensor = ToTensor()

fold_file = open(FOLD_PATH, 'rb')
folds = pickle.load(fold_file)

score_dict = {}
for i, (train_index, test_index) in enumerate(folds):
    train_data = np.array(dataset[train_index])
    test_data  = np.array(dataset[test_index])
    
    train_dataset = DifficultyFoldDataset(train_data, content_transform)
    test_dataset  = DifficultyFoldDataset(test_data, content_transform)
    
    vit_model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    regression_head = torch.nn.Sequential(
        torch.nn.Linear(in_features = 768 , out_features = 1),
        torch.nn.Sigmoid())
    vit_model.heads = regression_head
    vit_model = vit_model.to(device)  
    vit_model.train()
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)    
    
    criterion = torch.nn.MSELoss()
    optimizer = Adam(vit_model.parameters(), lr=0.0001)
    train_model(vit_model, train_dataloader,test_dataloader, optimizer, criterion)
    vit_model.eval()
    for item in test_dataset:
        image, score, path = item
        score = vit_model(image.unsqueeze(0).to(device))
        score_dict[path] = score


paths = train_df[['path']].values.tolist()
paths = [path[0] for path in paths]


scores = []
for path in paths:
    scores.append(float(score_dict[path].detach().cpu()))


result_df = pd.DataFrame({'path': paths, 'score': scores})
result_df.to_csv('/../../Results/vitregressor-{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric']),index=False)