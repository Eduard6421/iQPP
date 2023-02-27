#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from collections import Counter
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from ast import literal_eval
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis
import pickle
from joblib import dump, load
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
args = vars(parser.parse_args())


if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'


# In[2]:


query_embedding_df = pd.read_csv('../../Embeddings/{}/{}-query-features.csv'.format(EMBEDDINGS_FOLDER,args['dataset']))
db_embedding_df    = pd.read_csv('../../Embeddings/{}/{}-dataset-features.csv'.format(EMBEDDINGS_FOLDER,args['dataset']))



# In[3]:


def get_embeddings(query_embeddings, dataset_embeddings):
     
    num_images   =  len(dataset_embeddings)
    num_queries  =  len(query_embeddings)
    
    print('Generating query embedding array')
    
    q_embeddings = []
    
    for query_id in range(num_queries):
        image_name = query_embeddings.iloc[query_id]['image_name']
        query_embedding = np.array(literal_eval(query_embeddings.iloc[query_id]['embedding']))
        q_embeddings.append(query_embedding)

    q_embeddings = np.array(q_embeddings)
    
    
    print('Generating dataset embedding_array')
    
    db_embeddings = []
    
    for item_id in range(num_images):
        if(item_id%1000 == 0):
            print(item_id)
        image_embedding = literal_eval(dataset_embeddings.iloc[item_id]['embedding'])
        db_embeddings.append(image_embedding)
    
    db_embeddings = np.array(db_embeddings)    
    
    return q_embeddings,db_embeddings


# In[4]:


query_emb, db_emb = get_embeddings(query_embedding_df,db_embedding_df)


clustering = KMeans(150)
clustering.fit(db_emb)


x_train = db_emb
y_train = clustering.predict(db_emb)

clf = MLPClassifier(hidden_layer_sizes= (50,50),verbose=True, early_stopping=True,random_state=1).fit(x_train, y_train)


preds = clf.predict_proba(query_emb)
#dump(clf, '/notebooks/Kurtosis-And-Deviation/cnnimageretrieval-model-caltech101_700.joblib') 


scores = []
for i in range(preds.shape[0]):
    score = kurtosis(preds[i])
    scores.append(score)
    
scores = np.array(scores)
result_df = pd.DataFrame({'path': np.array(query_embedding_df['image_name']), 'score': scores})
result_df.to_csv('/notebooks/Results/kurtosis-{}-{}.csv'.format(args['method'],args['dataset']),index=False)

scores = []
for i in range(preds.shape[0]):
    score = np.std(preds[i])
    scores.append(score)
    
scores = np.array(scores)
result_df = pd.DataFrame({'path': np.array(query_embedding_df['image_name']), 'score': scores})
result_df.to_csv('/notebooks/Results/head-deviation-{}-{}.csv'.format(args['method'],args['dataset']),index=False)