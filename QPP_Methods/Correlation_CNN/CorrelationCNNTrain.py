#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset,DataLoader,random_split
from sklearn.model_selection import KFold
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from numpy import dot
from numpy.linalg import norm
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
parser.add_argument('--metric', required=True,help='Retrieval metric',choices=['ap','p@100'])
args = vars(parser.parse_args())


# -------------------------------------------------------- Global settings --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())

BATCH_SIZE = 32
NUM_EPOCHS = 100


if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'

# In[3]:


def cosine_similarity(x,y):
    return dot(x,y) / (norm(x) * norm(y))
def parse_input(content): 
    regex = r"\[([0-9\s,\-\.e]+)\]"
    items = re.findall(regex, content)
    parsed_input = np.array([np.fromstring(embed, sep=',') for embed in items]).astype(float)
    return parsed_input


# In[4]:


test_top_100_emb = '/notebooks/Embeddings/{}/{}-top-100-results-and-scores.csv'.format(EMBEDDINGS_FOLDER,args['dataset'])
test_scores_path = '/notebooks/Results/{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric'])

train_top_100_emb = '/notebooks/Embeddings/{}/{}_train-top-100-results-and-scores.csv'.format(EMBEDDINGS_FOLDER,args['dataset'])
train_scores_path = '/notebooks/Results/{}-{}_train-{}.csv'.format(EMBEDDINGS_FOLDER,args['dataset'])


# In[5]:


test_scores_df = pd.read_csv(test_scores_path)
train_scores_df = pd.read_csv(train_scores_path)


# In[6]:


test_embeddings_df = pd.read_csv(test_top_100_emb)
train_embeddings_df = pd.read_csv(train_top_100_emb)


# In[7]:


train_scores_df.shape


# In[8]:


test_inputs = []
for query_idx in range(len(test_embeddings_df)):
    if(query_idx % 50 == 0):
        print(query_idx)
    image_embeddings = test_embeddings_df['result_emb'].iloc[query_idx]
    image_embeddings = image_embeddings[1:-1]
    image_embeddings = parse_input(image_embeddings)
    test_inputs.append(image_embeddings)
test_inputs = np.array(test_inputs)


# In[9]:


train_inputs = []
for query_idx in range(len(train_embeddings_df)):
    if(query_idx % 50 == 0):
        print(query_idx)
    image_embeddings = train_embeddings_df['result_emb'].iloc[query_idx]
    image_embeddings = image_embeddings[1:-1]
    image_embeddings = parse_input(image_embeddings)
    train_inputs.append(image_embeddings)    
train_inputs = np.array(train_inputs)


# In[10]:


test_scores = test_scores_df['score'].to_numpy()
train_scores = train_scores_df['score'].to_numpy()


# In[11]:


test_input_maps = []
for query_idx in range(len(test_inputs)):
    result_elements = test_inputs[query_idx,:,:]
    similarity_matrix = np.zeros((100,100))
    for i in range(100):
        for j in range(i, 100):
            similarity_matrix[i][j] = similarity_matrix[j][i] = cosine_similarity(result_elements[i],result_elements[j])
    new_similarity_matrix = np.zeros((50,50))
    
    test_input_maps.append(similarity_matrix)
test_input_maps = np.array(test_input_maps)
test_input_maps = test_input_maps.reshape(len(test_inputs),1,100,100)


# In[12]:


test_paths = test_scores_df[['path']].to_numpy().squeeze(1)
test_dataset = np.array(list(zip(test_input_maps,test_scores, test_paths)))


# In[13]:


train_input_maps = []
for query_idx in range(len(train_inputs)):
    result_elements = train_inputs[query_idx,:,:]
    similarity_matrix = np.zeros((100,100))
    for i in range(100):
        for j in range(i, 100):
            similarity_matrix[i][j] = similarity_matrix[j][i] = cosine_similarity(result_elements[i],result_elements[j])
    new_similarity_matrix = np.zeros((50,50))
    
    train_input_maps.append(similarity_matrix)
train_input_maps = np.array(train_input_maps)
train_input_maps = train_input_maps.reshape(len(train_inputs),1,100,100)


# In[14]:


train_paths = train_scores_df[['path']].to_numpy().squeeze(1)
train_dataset = np.array(list(zip(train_input_maps,train_scores, train_paths)))


# In[15]:


class DifficultyDataset(Dataset):

    def __init__(self, data):
        self.correlation_matrices = data[:,0].tolist()
        self.scores = data[:,1]
        self.paths  = data[:,2].tolist()
    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):

        matrix = torch.tensor(self.correlation_matrices[idx])
        score = torch.tensor(float(self.scores[idx]))
        query_path = self.paths[idx]

        return (matrix, score, query_path)


# In[16]:


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


# In[17]:


def train_model(model, train_dataloader, validation_dataloader, optimizer , criterion):
    
    min_loss = 1000
    
    for i in range(NUM_EPOCHS):
        print("Epoch num {}/{}".format(i+1,NUM_EPOCHS))
        
        epoch_train_loss = 0
        
        for idx, data in enumerate(train_dataloader):
            #print("Batch num {}/{}".format(idx+1, len(train_dataloader)))
 
            (images,scores,img_paths) = data
    
            images = images.to(device,dtype=torch.float)
            scores = scores.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(scores, outputs)

            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_dataloader)
        
        epoch_validation_loss = compute_loss(model, validation_dataloader)

        print("Epoch train loss {}".format(epoch_train_loss))
        print("Epoch validation loss {}".format(epoch_validation_loss))


# In[18]:


class CNN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20 , 3, stride = 1)
        self.conv2 = nn.Conv2d(20, 50 , 3, stride = 1)
        self.conv3 = nn.Conv2d(50, 50 , 3, stride = 1)
        self.fc1   = nn.Linear(5000, 256)
        self.fc2   = nn.Linear(256, 1)
        
        self.max_pool   = nn.MaxPool2d(2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.leaky_relu(x)        
        
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.leaky_relu(x)
        
        x = x.reshape(-1,5000)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        
        x = self.fc2(x)
        
        return x


# In[19]:


train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)


# In[20]:


score_dict = {}
train_diff_dataset = DifficultyDataset(train_dataset)
test_diff_dataset  = DifficultyDataset(test_dataset)

total_size = len(train_diff_dataset)
train_size = int(0.8 * total_size)
validation_size = total_size - train_size

train_data, validation_data = torch.utils.data.random_split(train_diff_dataset, [train_size, validation_size])
    
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_diff_dataset, batch_size=BATCH_SIZE, shuffle=False)   
    
    
cnn_model = CNN_Network()
cnn_model = cnn_model.to(device)  
cnn_model.train()
    
criterion = torch.nn.MSELoss()
optimizer = Adam(cnn_model.parameters(), lr=0.0001)
train_model(cnn_model, train_dataloader, validation_dataloader , optimizer, criterion)
cnn_model.eval()
for item in test_dataset:
    image, score, path = item
    score = cnn_model(torch.tensor(image).unsqueeze(0).to(device,dtype=torch.float))
    score_dict[path] = score


scores = []
for path in test_paths:
    scores.append(float(score_dict[path].detach().cpu()))

result_df = pd.DataFrame({'path': test_paths, 'score': scores})
result_df.to_csv('/../../Results/sunetal-{}-{}-{}.csv'.format(args['method'],args['method'],args['metric']),index=False)