#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import skimage.io as sk
import scipy.stats as stats
import numpy as np
import torchvision

from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16,VGG16_Weights
from torch.utils.data import Dataset,DataLoader,random_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

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


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


CSV_FILEPATH = '/../../Referenced-Models/VSD_dataset.csv'
IMAGE_FOLDER = '/../../Datasets/{}/jpg'.format(DATASET_FOLDER)


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


# In[5]:


# Custom dataset class

class ImageDifficultyDataset(Dataset):
    def __init__(self, label_dir, image_dir, transform=None):
        df = pd.read_csv(label_dir, sep=',', header=None)
        df = df.rename(columns = {0:'img_name',1:'difficulty_score'})
        
        self.data = df
        self.rootDir = image_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transform])
        
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = self.rootDir + "/" + self.data['img_name'][idx] + '.jpg'
        image = Image.open(imagePath)
        if(self.transform):
            image = self.transform(image)
            
        score = self.data['difficulty_score'][idx]
            
        return image, score


# In[7]:


def get_input_output_tensors(dataloader):
    input_tensor_array = []
    output_tensor_array = []

    print('Extracting features....')
    with torch.no_grad():
        for idx,item in enumerate(dataloader):
            print('Batch {} / {}'.format(idx+1,len(dataloader)))
            (image_batches, score_batches) = item
            image_batches = image_batches.to(device)
            results = model(image_batches)
            
            input_tensor_array.append(results)
            output_tensor_array.append(score_batches)
            
    input_tensor_array = torch.stack(input_tensor_array[:-1],dim = 0)
    input_tensor_array = input_tensor_array.reshape(-1,4096)
    input_tensor_array = np.array(input_tensor_array.cpu())
    
    output_tensor_array = torch.stack(output_tensor_array[:-1],dim = 0)
    output_tensor_array = output_tensor_array.reshape(-1)
    output_tensor_array = np.array(output_tensor_array.cpu())
    
    return input_tensor_array,output_tensor_array
            

def generate_models():
    #Dataset Reads
    dataset = ImageDifficultyDataset(label_dir=CSV_FILEPATH,image_dir=IMAGE_FOLDER, transform =VGG16_Weights.DEFAULT.transforms() )

    train_size = int(0.80*int(len(dataset)))
    validation_size = int(0.10*int(len(dataset)))
    test_size  = len(dataset) - (train_size + validation_size)

    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths=[train_size,validation_size,test_size],generator=torch.Generator().manual_seed(420))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=300,shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    
    
    train_input_tensor,train_output_tensor = get_input_output_tensors(train_dataloader)
    validation_input_tensor,validation_output_tensor = get_input_output_tensors(validation_dataloader)
    test_input_tensor,test_output_tensor = get_input_output_tensors(test_dataloader)    
    
    
    scaler = preprocessing.StandardScaler().fit(train_input_tensor)
    normalized_train_features = scaler.transform(train_input_tensor)
    normalized_validation_features = scaler.transform(validation_input_tensor)
    normalized_test_features = scaler.transform(test_input_tensor)
    
    print('Starting training...')
    svr = SVR(degree=3, epsilon = 0.1)
    svr.fit(normalized_train_features,train_output_tensor)

    print('Train score: {}'.format(svr.score(normalized_train_features, train_output_tensor)))
    print('Validation score: {}'.format(svr.score(normalized_validation_features,validation_output_tensor )))
    print('Test score: {}'.format(svr.score(normalized_test_features, test_output_tensor)))

    return scaler, svr


# In[6]:


#model = VGGExtractor()


# In[8]:


#scaler,svr = generate_models()
#dump(scaler, 'scaler.joblib')
#dump(svr, 'svr.joblib')
#dump(model,'model.joblib')

