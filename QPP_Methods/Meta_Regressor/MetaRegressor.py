#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import NuSVR
from scipy import stats
from sklearn.model_selection import GridSearchCV


# In[4]:


# Objectness
# Ionescu et al.
# Denosising AE
# Masked AE
# Cluster Density
# ViT
# Unmasking
# Dispersion
# Adaptive Query Feedback
# Score Dispeprsion
# Metaregressor


# In[5]:


from typing import Iterable, Any
from itertools import product


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


# In[9]:


def collect_features(methods):
    items = []
    for i in range(len(methods[0])):
        feature_array = []
        for method in methods:
            feature_array.append(method.iloc[i]['score'])
        items.append(feature_array)
    return items


def classic_normalize(feature_vect):
    num_items    = feature_vect.shape[0]
    num_features = feature_vect.shape[1]

    for feature_idx in range(num_features):
        min_value = np.min(feature_vect[:,feature_idx])
        max_value = np.max(feature_vect[:,feature_idx])
        feature_vect[:,feature_idx]  -= min_value
        feature_vect[:,feature_idx] /= (max_value - min_value)
    
    return feature_vect
        

def run_meta(DATASET_NAME, RETRIEVAL_METHOD, METRIC , LEVEL, FOLD_PATH):
    ground_truth = pd.read_csv('../../Results/{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,METRIC))
    objectness            = pd.read_csv('../../Results/objectness-results-{}.csv'.format(DATASET_NAME))
    ietal                 = pd.read_csv('../../Results/ionescu-et-al-{}.csv'.format(DATASET_NAME))
    denoising_autoencoder = pd.read_csv('../../Results/denoising-autoencoder-{}.csv'.format(DATASET_NAME))
    masked_autoencoder    = pd.read_csv('../../Results/masked-autoencoder-{}.csv'.format(DATASET_NAME))
    cluster_density       = pd.read_csv('../../Results/preretrieval-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME))
    vit_regressor         = pd.read_csv('../../Results/vitregressor-{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,METRIC))
    unmasking             = pd.read_csv('../../Results/unmasking-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME))
    emb_dispersion        = pd.read_csv('../../Results/postretrieval-{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,LEVEL))
    score_dispersion      = pd.read_csv('../../Results/score-dispersion-{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,LEVEL))
    adaptive_qf           = pd.read_csv('../../Results/adaptivequery-{}-{}-100.csv'.format(RETRIEVAL_METHOD,DATASET_NAME))    
    sunetal               = pd.read_csv('../../Results/sunetal-{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,METRIC))
    
    
    methods = [
    objectness,
    ietal,
    denoising_autoencoder,
    masked_autoencoder,
    cluster_density,
    vit_regressor,
    unmasking,
    emb_dispersion,
    score_dispersion,
    adaptive_qf,
    sunetal
    ]
    
    
    input_features  = np.array(collect_features(methods))
    paths  = np.array(ground_truth['path'].tolist())
    outputs = np.array(ground_truth['score'].tolist())  
    
    fold_file = open(FOLD_PATH, 'rb')
    folds = pickle.load(fold_file)    
    
    score_dict = {}
    
    input_features = classic_normalize(input_features)
    outputs        = classic_normalize(outputs.reshape(-1,1))
    
    C, nu, best_score = hyperparameter_search(folds, input_features, outputs)
    
    print("{} - {} - {} :   C = {} and nu = {} with score {}".format(DATASET_NAME, RETRIEVAL_METHOD, METRIC, C, nu, best_score))
    
    for i, (train_index, test_index) in enumerate(folds):
        
        train_data,train_scores = input_features[train_index],outputs[train_index]    
        test_data,test_scores  = input_features[test_index],outputs[test_index]
        test_paths = paths[test_index]
        

        model = NuSVR(kernel='rbf',nu = nu,C = C)
        model.fit(train_data,train_scores.squeeze(1))
        score = model.predict(test_data)
        
        for path_id, path in enumerate(test_paths):
            score_dict[path] = score[path_id]
        
    scores = []
    for path in paths:
        scores.append(score_dict[path])        
        
    result_path = '../../Results/metaregressor-{}-{}-{}.csv'.format(RETRIEVAL_METHOD,DATASET_NAME,METRIC)
    result_df = pd.DataFrame({'path': paths, 'score': scores})
    result_df.to_csv(result_path,index=False)
    
def hyperparameter_search(folds, input_features, outputs):
    
    best_corr = 0
    best_C = None
    best_nu = None
    
    for C in [0.1,1,10,100,1000]:
        for nu in np.arange(0.1,1,0.1):
            #print("C : {}, nu : {}".format(C,nu))
            
            all_predictions = np.zeros(len(outputs))
            
            for i, (train_index, test_index) in enumerate(folds):

                train_data,train_scores = input_features[train_index],outputs[train_index]    
                test_data,test_scores  = input_features[test_index],outputs[test_index]                
    
                model = NuSVR(kernel='rbf',nu = nu,C = C)
                model.fit(train_data,train_scores.squeeze(1))
            
                predictions = model.predict(test_data)
                
                all_predictions[test_index] = predictions

            tau, p_value = stats.kendalltau(all_predictions, outputs)
            
            if(tau > best_corr):
                best_corr = tau
                best_C = C
                best_nu = nu
    return best_C, best_nu, best_corr



DATASET_NAMES = ['roxford5k','rparis6k','caltech101_700','pascalvoc_700_medium']
RETRIEVAL_METHODS = ['cnnimageretrieval','deepretrieval']
METRICS = ['ap']
LEVELS = ['100']


for dataset_name in DATASET_NAMES:
    for method in RETRIEVAL_METHODS:
        for metric in METRICS:
            for level in LEVELS:
                folder_path = '../../Folds/{}-folds.pkl'.format(dataset_name)
                run_meta(dataset_name, method, metric, level,folder_path)