#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------- Imports --------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn


# In[1]:


# -------------------------------------------------------- Model definition  --------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, num_channels):
        
        super(Encoder,self).__init__()
        
        self.conv1 = nn.Conv2d(3               , num_channels * 1,  2, stride = 2, padding = 0)
        self.conv2 = nn.Conv2d(num_channels    , num_channels * 2,  2, stride = 2, padding = 0)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4,  2, stride = 2, padding = 0)
        self.conv4 = nn.Conv2d(num_channels * 4, num_channels * 8,  2, stride = 2, padding = 0)
        
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(num_channels * 1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.bn3 = nn.BatchNorm2d(num_channels * 4)
        self.bn4 = nn.BatchNorm2d(num_channels * 8)
        
    
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, num_channels):
        
        super(Decoder, self).__init__()
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv2d(num_channels * 8,  num_channels * 8, 2, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(num_channels * 8,  num_channels * 4, 2, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(num_channels * 4,  num_channels * 2, 2, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(num_channels * 2,  num_channels * 1, 2, stride = 1, padding = 0)
        self.conv5 = nn.Conv2d(num_channels * 1,                 3, 3, stride = 1, padding = 0)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.bn1 = nn.BatchNorm2d(num_channels * 8)
        self.bn2 = nn.BatchNorm2d(num_channels * 4)
        self.bn3 = nn.BatchNorm2d(num_channels * 2)
        self.bn4 = nn.BatchNorm2d(num_channels * 1)
        self.bn5 = nn.BatchNorm2d(3)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.upsample(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sigmoid(x)
        
        
        return x    

    
class DenoisingAutoencoder(nn.Module):
    def __init__(self, num_channels):
        
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = Encoder(num_channels)
        self.decoder = Decoder(num_channels) 
        
    def forward(self, x):
        z = self.encoder(x) # Latent expression
        y = self.decoder(z) # Decoding
        return y 
    