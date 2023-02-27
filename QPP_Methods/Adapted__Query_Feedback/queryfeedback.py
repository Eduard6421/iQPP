#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,help='Dataset on which you want to train the model',choices=['roxford5k','rparis6k','pascalvoc_700_medium','caltech101_700'])
parser.add_argument('--method', required=True,help='Retrieval method which you want to analyse',choices=['cnnimageretrieval','deepretrieval'])
parser.add_argument('--metric', required=True,help='Retrieval metric',choices=['ap','p@100'])
args = vars(parser.parse_args())

if(args['method'] == "cnnimageretrieval"):
    EMBEDDINGS_FOLDER = 'CNN_Image_Retrieval'
else:
    EMBEDDINGS_FOLDER = 'DEEP_Image_Retrieval'

result_embeddings = pd.read_csv('../../Embeddings/{}/{}-top-100-results-and-scores.csv'.format(args['method'],args['dataset']))
feedback_embeddings = pd.read_csv('../../Embeddings/{}/{}-drift-top-100-results-and-scores.csv'.format(args['method'],args['dataset']))

def text_parse_paths(text):
    new_text = ""
    for letter in text:
        if(not(letter in(['[',']',',']))):
            new_text+=letter
    new_text= np.array(new_text.split(' '))
    return new_text


def parse_array(arr):
    new_arr = []
    for i in range(0,100):
        temp_arr = arr[i*2048:(i+1)*2048]
        new_arr.append(temp_arr)
    
    new_arr = np.array(new_arr)
    return new_arr


def intersection_over_union(items):
    intersection = np.array(items[0])
    union = np.array(items[0])
    
    for i in range(1, len(items)):
        intersection = np.intersect1d(intersection , items[i])
        union = np.union1d(union, items[i])
        
    return len(intersection) / len(union)


scores = []
for query_idx in range(result_embeddings.shape[0]):
    query_path = result_embeddings.iloc[query_idx]['query_path']
    orig_cbir_results     = text_parse_paths(result_embeddings.iloc[query_idx]['results_path'])
    adaptive_cbir_results = text_parse_paths(feedback_embeddings.iloc[query_idx]['results_path'])
    iou_score = intersection_over_union([orig_cbir_results,adaptive_cbir_results])
    scores.append(iou_score)


result_df = pd.DataFrame({'path': np.array(result_embeddings['query_path']), 'score': scores})

result_df.to_csv('/../../Datasets/Results/adaptivequery-{}-{}.csv'.format(args['method'],args['dataset']),index=False)

