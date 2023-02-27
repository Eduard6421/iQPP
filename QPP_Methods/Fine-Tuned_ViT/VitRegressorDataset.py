#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from sklearn.model_selection import KFold
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


TRAIN_PERCENT = 0.8
VALIDATION_PERCENT = 0.1
BATCH_SIZE = 64
RESHAPE_SIZE = 512
NUM_EPOCHS = 25
LR = 0.0001
#LR = 0.00001



# -------------------------------------------------------- Custom Transforms --------------------------------------------------------------

class ExpandDimension(object):
    def __call__(self, sample):
        if(sample.shape[0] == 1):
            sample = sample.repeat(3,1,1)
        return sample



content_transform = Compose([ViT_B_32_Weights.IMAGENET1K_V1.transforms()])
expand_dims_transform = Compose([ToTensor(),ExpandDimension()])



class DifficultyDataset(Dataset):

    def __init__(self, csv_file_path, transform=None):
        self.scores_df = pd.read_csv(csv_file_path)
        self.image_paths = self.scores_df['path'].tolist()
        self.scores = self.scores_df['score'].tolist()

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if(transform):
            image = expand_dims_transform(image)
            image = transform(image)
        
        score = torch.tensor(self.scores[idx])

        return image,score,img_path



train_dataset = DifficultyDataset('/../../Results/{}-{}_train-{}.csv'.format(args['dataset'],args['method'],args['metric']), transform=content_transform)
test_dataset = DifficultyDataset('/../../Results/{}-{}-{}.csv'.format(args['dataset'],args['method'],args['metric']), transform=content_transform)
gt_file = pd.read_csv('/../../Results/{}-{}-{}.csv'.format(args['dataset'],args['method'],args['metric']))



vit_model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
regression_head = torch.nn.Sequential(
    torch.nn.Linear(in_features = 768 , out_features = 1),
    torch.nn.Sigmoid())
vit_model.heads = regression_head

transform = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
vit_model = vit_model.to(device)



train_size = int(TRAIN_PERCENT*int(len(train_dataset)))
validation_size = len(train_dataset) - train_size

train_dataset, validation_dataset = random_split(train_dataset, lengths=[train_size,validation_size],generator=torch.Generator().manual_seed(420))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)



test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True)



criterion = torch.nn.MSELoss()
optimizer = Adam(vit_model.parameters(), lr=LR)



def compute_loss(model , loader):
    
    total_loss = 0.0
    
    with torch.no_grad():
        for idx,data in enumerate(loader):
            images, scores, path = data
            
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)
            
            outputs = model(images)
            
            loss = criterion(outputs, scores)
            total_loss += loss.item()
    
    total_loss /= len(loader)
    
    return total_loss


def train_model(model, train_dataloader, validation_dataloader, optimizer , criterion):
    
    max_loss = 1000
    for i in range(NUM_EPOCHS):
        print("Epoch num {}/{}".format(i+1,NUM_EPOCHS))
        
        epoch_train_loss = 0
        
        for idx, data in enumerate(train_dataloader):
            #print("Batch num {}/{}".format(idx+1, len(train_dataloader)))
 
            (images,scores,path) = data
    
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(scores, outputs)
            

            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_dataloader)
        validation_loss = compute_loss(model, validation_dataloader)
        
        print("Epoch train loss {}".format(epoch_train_loss))
        print("Epoch validation loss {}".format(validation_loss))



vit_model.train()
train_model(vit_model, train_dataloader,validation_dataloader,optimizer, criterion)


score_dict = {}
vit_model.eval()
for (image, score , path) in test_dataloader:
    image = image.to(device)
    output = vit_model(image)
    score_dict[path[0]] = output.item()


paths = gt_file.values.tolist()
paths = [path[0] for path in paths]


scores = []
for path in paths:
    scores.append(score_dict[path])


result_df = pd.DataFrame({'path': paths, 'score': scores})
result_df.to_csv('/../../Results/vitregressor-{}-{}-{}.csv'.format(args['dataset'],args['method'],args['metric']),index=False)
