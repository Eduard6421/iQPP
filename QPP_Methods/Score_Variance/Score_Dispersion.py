#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples
from ast import literal_eval


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


top_100_query_path = '../../Embeddings/{}/{}-top-100-results-and-scores.csv'.format(DATASET_FOLDER,args['dataset'])


# In[3]:


top_100_results = pd.read_csv(top_100_query_path)


# In[4]:


title = np.array(top_100_results[['query_path']].values.tolist())[:1400]


def parse_scores(content):
    content = content[1:]
    content = content[:-1]
    content = content.split()
    content = np.array(content).astype(float)
    return content



top_100_results.shape



cibr_scores = []
    
for item_id in range(top_100_results.shape[0]):
    scores = top_100_results.iloc[item_id]['scores']
    scores = parse_scores(scores)
    #Takign only first 10 scores
    scores = scores[:100]
    difficulty_score = np.var(scores)
    cibr_scores.append(difficulty_score)
    
cibr_scores = np.array(cibr_scores) 


cibr_scores.shape



title = title.flatten()


df = pd.DataFrame({'path': title, 'score': cibr_scores})

df.to_csv('../../Results/score-dispersion-{}-{}-100.csv'.format(args['dataset'],args['metric']))