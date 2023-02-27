#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import pickle
from pathlib import Path
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



results_embeddings = pd.read_csv('../../Embeddings/{}/{}-top-100-results-and-scores.csv'.format(args['method'],args['dataset']))




def text_parser(text):
    new_text = ""
    for letter in text:
        if(not(letter in(['[',']',',']))):
            new_text+=letter
    new_text= np.array(new_text.split(' '))
    new_text= new_text.astype(float)
    return new_text


def parse_array(arr):
    new_arr = []
    for i in range(0,100):
        temp_arr = arr[i*2048:(i+1)*2048]
        new_arr.append(temp_arr)
    
    new_arr = np.array(new_arr)
    return new_arr

def get_closest(searched, embedding_list):
    smallest_distance = np.linalg.norm(searched-embedding_list[0,:])
    current_closest_idx = 0

    for embedding_idx in range(1, embedding_list.shape[0]):
        distance = np.linalg.norm(searched - embedding_list[embedding_idx,:])
        if(distance < smallest_distance):
            smallest_distance = distance
            current_closest_idx = embedding_idx
    
    return current_closest_idx



closest_queries = []
for query_idx in range(len(results_embeddings)):
    returned_embeddings = results_embeddings.iloc[query_idx]['result_emb']
    returned_embeddings = parse_array(text_parser(returned_embeddings))
    avg_embed = np.mean(returned_embeddings, axis = 0)
    closest_idx = get_closest(avg_embed, returned_embeddings)
    closest_queries.append(closest_idx)
    #print(query_idx+ 1)



def text_parse_paths(text):
    new_text = ""
    for letter in text:
        if(not(letter in(['[',']',',']))):
            new_text+=letter
    new_text= np.array(new_text.split(' '))
    return new_text



new_query_names  = []
for query_idx in range(len(results_embeddings)):
    image_name = text_parse_paths(results_embeddings.iloc[query_idx]['results_path'])[closest_queries[query_idx]]
    new_query_names.append(image_name)

new_query_names = [ Path(name).stem for name in new_query_names]



file = open('/../../Datasets/{}/gnd_{}.pkl'.format(DATASET_FOLDER),'rb')
new_gt_file = open('/../../Datasets/gnd_{}_drift_cnn.pkl'.format(DATASET_FOLDER),'wb')
gt_file = pickle.load(file)



new_gts = [{
'bbx' : None,
'easy' : [0],
'hard' : [],
'junk' : [],
'class' : '',
} for name in new_query_names]



new_gt_content = {
    'qimlist' : new_query_names,
    'imlist' : gt_file['imlist'],
    'gnd': new_gts}


pickle.dump(new_gt_content, new_gt_file, protocol = pickle.HIGHEST_PROTOCOL)


file.close()
new_gt_file.close()

