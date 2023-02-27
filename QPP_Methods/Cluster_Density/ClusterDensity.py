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
from scipy.spatial import distance
from scipy import stats

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


cnn_dataset_path = '../../Embeddings/{}/{}-dataset-features.csv'.format(EMBEDDINGS_FOLDER, args['dataset'])
cnn_query_path = '../../Embeddings/{}/{}-query-features.csv'.format(EMBEDDINGS_FOLDER, args['dataset'])


df_db = pd.read_csv(cnn_dataset_path)
df_query = pd.read_csv(cnn_query_path)


db_embeddings = []
    
for item_id in range(len(df_db)):
    image_embedding = literal_eval(df_db.iloc[item_id]['embedding'])
    db_embeddings.append(image_embedding)
    
db_embeddings = np.array(db_embeddings) 


df_queries = []
    
for item_id in range(len(df_query)):
    image_embedding = literal_eval(df_query.iloc[item_id]['embedding'])
    df_queries.append(image_embedding)
    
df_queries = np.array(df_queries) 


title = np.array(df_query[['image_name']].values.tolist())
title = title.flatten()


def train_clustering():
    num_clusters = 150
    model = KMeans(num_clusters)
    model.fit(db_embeddings)
    cluster_score = np.zeros(num_clusters)

    db_preds = model.predict(db_embeddings)
    q_preds  = model.predict(df_queries)

    clusters_lengths = []
    clusters_mean_distance = []

    for i in range(num_clusters):
        cluster_center = model.cluster_centers_[i]
        cluster_elements = np.array([ db_embeddings[idx] for idx in range(len(db_embeddings)) if db_preds[idx] == i])
        num_elements = len(cluster_elements)

        mean_distance = 0 

        for embedding in cluster_elements:
            distance_to_center = distance.euclidean(embedding, cluster_center)
            mean_distance += distance_to_center

        # Presume that for the query we will have another item. So we divide to num_elements + 1 as we want to include the query image into the mean distance
        mean_distance /= (num_elements)

        clusters_mean_distance.append(mean_distance)
        # Trick. Add one more to the number of elements. presume you have found another!
        clusters_lengths.append(num_elements)
    difficulty_score = []

    for i in range(len(q_preds)):
        query_emb = df_queries[i]
        cluster_prediction = q_preds[i]

        num_elements  = clusters_lengths[cluster_prediction]
        mean_distance = clusters_mean_distance[cluster_prediction] + distance.euclidean(query_emb, model.cluster_centers_[cluster_prediction])
        image_score = num_elements / mean_distance

        difficulty_score.append(image_score)
    df = pd.DataFrame({'path': title, 'score': difficulty_score})
    cnnretrieval = pd.read_csv('../../Results/{}-{}-{}.csv'.format(args['method'],args['dataset'],args['metric']))
    joined_df_cnn = cnnretrieval.merge(df, on='path')
    results_clustering   = joined_df_cnn[['score_x','score_y']]
    score_x = results_clustering['score_x'].tolist()
    score_y = results_clustering['score_y'].tolist()
    pearson, pearson_p_value = stats.pearsonr(score_x,score_y)
    tau, tau_p_value = stats.kendalltau(score_x, score_y)

    return pearson,pearson_p_value,tau,tau_p_value


pearsons = []
pearson_p_values = []
taus = []
tau_p_values = []

for i in range(5):
    print(i)
    pearson,pearson_p_value,tau,tau_p_value = train_clustering()
    pearsons.append(pearson)
    pearson_p_values.append(pearson_p_value)
    taus.append(tau)
    tau_p_values.append(tau_p_value)


print("pearson: {} / p-value: {}".format(np.mean(pearsons),np.mean(pearson_p_values)))
print("tau: {}  / p-value: {}".format(np.mean(taus), np.mean(tau_p_values)))