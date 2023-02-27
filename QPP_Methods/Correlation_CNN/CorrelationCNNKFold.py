import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset,DataLoader,random_split
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

EMBEDDINGS_FOLDER = None

if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'


# -------------------------------------------------------- Global settings --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}".format())

BATCH_SIZE = 70
NUM_EPOCHS = 500



top_100_emb = '/../../Embeddings/{}/{}-top-100-results-and-scores.csv'.format(args['method'],args['metric'])
scores_path = '/../../Results/{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric'])
FOLD_PATH = '/../../Folds/{}-folds.pkl'.format(args['dataset'])


# In[4]:


embeddings_df = pd.read_csv(top_100_emb)
scores_df = pd.read_csv(scores_path)


# In[5]:


def cosine_similarity(x,y):
    return dot(x,y) / (norm(x) * norm(y))

def parse_input(content): 
    regex = r"\[([0-9\s,\-\.e]+)\]"
    items = re.findall(regex, content)
    parsed_input = np.array([np.fromstring(embed, sep=',') for embed in items]).astype(float)
    return parsed_input


# In[6]:


inputs = []
for query_idx in range(len(embeddings_df)):
    if(query_idx % 50 == 0):
        print(query_idx)
    image_embeddings = embeddings_df['result_emb'].iloc[query_idx]
    image_embeddings = image_embeddings[1:-1]
    image_embeddings = parse_input(image_embeddings)
    inputs.append(image_embeddings)
    
inputs = np.array(inputs)
scores = scores_df['score'].to_numpy()


# In[7]:


input_maps = []
for query_idx in range(len(inputs)):
    result_elements = inputs[query_idx,:,:]
    similarity_matrix = np.zeros((100,100))
    for i in range(100):
        for j in range(i, 100):
            similarity_matrix[i][j] = similarity_matrix[j][i] = cosine_similarity(result_elements[i],result_elements[j])
    input_maps.append(similarity_matrix)
input_maps = np.array(input_maps)


# In[8]:


input_maps = input_maps.reshape(len(inputs),1,100,100)


# In[9]:


paths = scores_df[['path']].to_numpy().squeeze(1)


# In[10]:


dataset = np.array(list(zip(input_maps,scores, paths)))


# In[11]:


fold_file = open(FOLD_PATH, 'rb')
folds = pickle.load(fold_file)


# In[12]:


class DifficultyFoldDataset(Dataset):

    def __init__(self, data):
        self.correlation_matrices = data[:,0]
        self.scores = data[:,1]
        self.paths  = data[:,2]

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):

        matrix = torch.tensor(self.correlation_matrices[idx])
        score = torch.tensor(float(self.scores[idx]))
        query_path = self.paths[idx]

        return (matrix, score, query_path)


# In[13]:


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


# In[14]:


import matplotlib.pyplot as plt

def train_model(model, train_dataloader, test_dataloader, optimizer , criterion):
    
    min_loss = 1000
    for i in range(NUM_EPOCHS):
        
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
            
        epoch_test_loss = compute_loss(model, test_dataloader)

        epoch_train_loss /= len(train_dataloader)
        if((i+1)%50== 0):
            print("Epoch num {}/{}".format(i+1,NUM_EPOCHS))
            print("Epoch train loss {}".format(epoch_train_loss))
            print("Epoch test loss {}".format(epoch_test_loss))


# In[15]:


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


# In[16]:


score_dict = {}
for i, (train_index, test_index) in enumerate(folds):
    train_data = np.array(dataset[train_index])
    test_data  = np.array(dataset[test_index])
    
    train_dataset = DifficultyFoldDataset(train_data)
    test_dataset  = DifficultyFoldDataset(test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)    
    
    
    
    cnn_model = CNN_Network()
    
    cnn_model = cnn_model.to(device)  
    cnn_model.train()
    
    criterion = torch.nn.MSELoss()
    optimizer = Adam(cnn_model.parameters(), lr=0.0005)
    train_model(cnn_model, train_dataloader, test_dataloader, optimizer, criterion)
    cnn_model.eval()
    
    test_loss = 0
    for item in test_dataset:
        image, true_score, path = item
        score = cnn_model(image.unsqueeze(0).to(device,dtype=torch.float))
        score_dict[path] = score


# In[17]:


scores = []
for path in paths:
    scores.append(float(score_dict[path].detach().cpu()))


# In[18]:


result_df = pd.DataFrame({'path': paths, 'score': scores})
result_df.to_csv('/../../Results/sunetal-{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric']),index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




