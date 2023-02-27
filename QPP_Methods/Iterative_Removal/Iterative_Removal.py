#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import pickle
from collections import Counter
from scipy.stats import kendalltau
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


query_embedding_df = pd.read_csv('../../Embeddings/{}/{}-query-features.csv'.format(DATASET_FOLDER, args['dataset']))
db_embedding_df    = pd.read_csv('../../Embeddings/{}/{}-dataset-features.csv'.format(DATASET_FOLDER, args['dataset']))


# In[4]:


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


# In[5]:


def intersection_over_union(items):
    intersection = np.array(items[0])
    union = np.array(items[0])
    
    for i in range(1, len(items)):
        intersection = np.intersect1d(intersection , items[i])
        union = np.union1d(union, items[i])
        
    return len(intersection) / len(union)


# In[6]:


def unmask(query_emb, db_emb, num_removed_features = 100, num_iterations = 10, num_top_items = 10):
    
    num_images   =  len(db_emb)
    num_queries  =  len(query_emb)
    
    scores = []
    
    
    for q_idx in range(num_queries):
        print("num query: {}".format(q_idx))
        temp_db_emb = db_emb.copy()
        
        current_query = query_emb[q_idx,:].copy()
        
        query_results = []
        
        for i in range(num_iterations):
            hadamard_prod = np.multiply(temp_db_emb, current_query)
            dot_prod = np.dot(current_query,temp_db_emb.T)
            
            # Retrieve the top k results and their indexes.
            maxed_dot_prod_idx = np.argsort(-dot_prod)
            top_k_idx = maxed_dot_prod_idx[:num_top_items]
            query_results.append(top_k_idx)
            
            #subselect only the top K results to compute the correlation
            sub_hadamard = np.take(hadamard_prod, top_k_idx,axis = 0)
            
            # Making a tuple of (feature_index, correlation_score)
            feature_correlations = []
            
            # Compute the correlation of each feature with the ranking
            for feature_idx in range(sub_hadamard.shape[1]):
                feature_vect = sub_hadamard[:,feature_idx]
                kendall_corr, p_value = kendalltau(feature_vect, np.arange(num_top_items))
                feature_correlations.append((feature_idx, kendall_corr))
                                          
            #sort to get most correlated features
            feature_correlations.sort(key=lambda x : -x[1])
            
            #get the most correlated feature indexes
            remove_feature_idx = [feature_tuple[0] for feature_tuple in feature_correlations]
            remove_feature_idx = remove_feature_idx[:num_removed_features]
            
            # remove the features from both the query and the database
            current_query = np.delete(current_query, remove_feature_idx)
            temp_db_emb = np.delete(temp_db_emb,remove_feature_idx, axis = 1)
        
        query_results = np.array(query_results)
        scores.append(intersection_over_union(query_results))
    
    scores = np.array(scores)
    
    return scores


# In[7]:


from itertools import product 
def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield(dict(zip(parameters.keys(),params)))


# In[8]:


# original is 100 / 15 / 100
params = {
'num_removed_features' : [100],
'num_iterations' : [15],
'num_top_items' : [100]}


# In[9]:


query_emb, db_emb = get_embeddings(query_embedding_df,db_embedding_df)


# In[10]:


for args in grid_parameters(params):
    print(args)
    scores = unmask(query_emb, db_emb, **args)
    unmask_df = pd.DataFrame({'path' : query_embedding_df['image_name'] ,'score': scores})
    unmask_df.to_csv('../../Results/unmasking-{}-{}.csv'.format(DATASET_FOLDER, args['dataset']),index=False)
    #run /notebooks/Comparisons/Compare-Unmasking.ipynb

