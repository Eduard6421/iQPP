#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
parser.add_argument('--metric', required=True,help='Retrieval metric',choices=['ap','p@100'])
args = vars(parser.parse_args())

from ast import literal_eval


if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'

# In[ ]:


top_100_query_path = '../../Embeddings/{}/{}-top-100-results-and-scores.csv'.format(EMBEDDINGS_FOLDER,args['dataset'])


# In[ ]:


top_100_results = pd.read_csv(top_100_query_path)


# In[ ]:


title = np.array(top_100_results[['query_path']].values.tolist())


# In[ ]:


db_embeddings = []

def parse_array(arr):
    new_arr = []
    for i in range(0,100):
        temp_arr = arr[i*2048:(i+1)*2048]
        new_arr.append(temp_arr)
    
    new_arr = np.array(new_arr)
    return new_arr
    

def text_parser(text):
    new_text = ""
    for letter in text:
        if(not(letter in(['[',']',',']))):
            new_text+=letter
    new_text= np.array(new_text.split(' '))
    new_text= new_text.astype(float)
    return new_text
    
for item_id in range(len(title)):
    print(item_id)
    current_emb = top_100_results.iloc[item_id]['result_emb']
    image_embedding = text_parser(current_emb)#literal_eval(top_100_results.iloc[item_id]['result_emb'])
    image_embedding = parse_array(image_embedding)
    db_embeddings.append(image_embedding)
    if(image_embedding.shape[1] != 2048):
        print('error in parsing')
        break
    
db_embeddings = np.array(db_embeddings) 


# In[ ]:


from scipy.spatial import distance


difficulty_score = []

for i in range(len(title)):
    print(i)
    data = db_embeddings[i,:10,:]
    model = KMeans(1)
    model.fit(data)
    
    cluster_center = model.cluster_centers_[0]
    
    total_distance = 0
    # Sum of distances between embedding and centroid
    for embedding in data:
        distance_to_center = distance.euclidean(embedding, cluster_center)
        total_distance += distance_to_center
        
    total_distance /= len(db_embeddings[i])
        
    
    difficulty_score.append(total_distance)


# In[ ]:


title = title.flatten()


# In[ ]:


df = pd.DataFrame({'path': title, 'score': difficulty_score})


# In[ ]:


df.to_csv('../../Results/postretrieval-{}-{}-100.csv'.format(args['method'],args['dataset']))


# In[ ]:


from scipy.spatial import distance


difficulty_score = []

for i in range(len(title)):
    print(i)
    data = db_embeddings[i,:10,:]
    model = KMeans(1)
    model.fit(data)
    
    cluster_center = model.cluster_centers_[0]
    
    total_distance = 0
    # Sum of distances between embedding and centroid
    for embedding in data:
        distance_to_center = distance.euclidean(embedding, cluster_center)
        total_distance += distance_to_center
        
    total_distance /= len(db_embeddings[i])
        
    
    difficulty_score.append(total_distance)


# In[ ]:


df = pd.DataFrame({'path': title, 'score': difficulty_score})


# In[ ]:


df.to_csv('../../Results/postretrieval-{}-{}-100.csv'.format(args['method'],args['dataset']))

