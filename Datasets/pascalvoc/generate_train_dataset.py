import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from pathlib import Path
from math import trunc
from functools import reduce

ANNOTATION_FOLDER_PATH = '.\Annotations'
IMAGE_SETS_FOLDER_PATH = '.\JPEGImages'

# Structure is
#
# class_name: {
#   difficulty_level : [id_image_1, id_image2,id_image3],
# }
#
##

dataset = {

}


related_to_categories = {

}


def read_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()

    file_name = Path(root.find('filename').text).stem

    objects = root.findall('object')
    for object in objects:
        try:

            class_name = object.find('name').text
            estimated_difficulty = 'easy'
            files_per_difficulty = dataset.setdefault(class_name, {})
            image_list = files_per_difficulty.setdefault(
                estimated_difficulty, [])


            related_classes = related_to_categories.setdefault(file_name, [])

            if(file_name not in image_list):
                image_list.append(file_name)

            if(class_name not in related_classes):
                related_classes.append(class_name)

            related_to_categories[file_name] = related_classes

            files_per_difficulty[estimated_difficulty] = image_list
            dataset[class_name] = files_per_difficulty
        except Exception as e:
            print("{} - has an error {}".format(file_name, e))



def list_annotations(folder_path):
    files = os.listdir(folder_path)
    for file_name in files:
        full_file_path = os.path.join(folder_path, file_name)
        read_annotations(full_file_path)



def create_cnn_cfg(avoid_queries = []):

    # 700 for easy queries / 700 hard queries
    num_queries = 700
    num_classes = len(dataset.keys())
    classes_extra = num_queries % num_classes
    num_images    = num_queries // num_classes


    # This is the distribution of the queries we want
    num_images_for_each = np.full((num_classes), num_images)
    num_images_for_each[:classes_extra] += 1


    queries = []
    tqlist = []
    qimlist = []

    # Go through all different classes and make sure to get an even distribution
    idx = 0
    while idx < len(dataset.keys()):
        key = list(dataset.keys())[idx]
        restricted_image_list = np.array(dataset[key]['easy'])
        index_array = np.arange(len(restricted_image_list))
        np.random.shuffle(index_array)
        
        zipped_queries = []
        i = 0
        while(num_images_for_each[idx] > 0):

            item = None
            try:
                item = restricted_image_list[index_array[i]]

                if((len(queries)+ len(zipped_queries))%2 == 1):
                    if (len(related_to_categories[item]) == 1):
                        i+=1 
                        continue
                    all_with_at_least_one = []


                    relevant_classes = related_to_categories[item]

                    print(relevant_classes)

                    for relevant_class in relevant_classes:
                        current_related = []
                        current_easy = dataset[relevant_class]['easy']
                        for relevant_image in current_easy:
                            if(relevant_image in qimlist):
                                continue
                            else:
                                current_related.append(relevant_image)
                                all_with_at_least_one.append(current_related)
                    # If we at least a few gt available
                    related_gt = list(reduce(set.intersection, [set(item) for item in all_with_at_least_one]))

                    print(len(related_gt))

                    if(len(related_gt) < 5):
                        print('eliminated')
                        i+=1 
                        continue

                if(item in tqlist or item in avoid_queries):
                    i += 1
                    continue
                else:
                    tqlist.append(item)
                    zipped_queries.append((key,item))
                    num_images_for_each[idx] -= 1
                    i += 1

            except:
                print('no more possible matches for this one')
                items_for_others = num_images_for_each[idx]
                num_images_for_each[idx] = -1
                idx = -1

                compat_indexes = []
                for temp_idx in range(len(dataset.keys())):
                    if(num_images_for_each[temp_idx] != -1):
                        compat_indexes.append(temp_idx)

                
                get_num_for_each_compat = int(len(compat_indexes) / items_for_others)
                get_remainder_of_one   = len(compat_indexes) % items_for_others

                for compat_idx in compat_indexes:
                    num_images_for_each[compat_idx] = num_images_for_each[compat_idx] + get_num_for_each_compat

                num_images_for_each[compat_indexes[0]] = num_images_for_each[compat_indexes[0]] + get_remainder_of_one
                break

        queries += zipped_queries

        qimlist += [Path(pair[1]).stem for pair in zipped_queries]

        idx+=1


    print(len(qimlist))
    qimlist = qimlist[:700]

    # remove query images from image list
    imlist = [Path(file_name).stem for file_name in os.listdir(IMAGE_SETS_FOLDER_PATH) if Path(file_name).stem not in qimlist]


    gnd = []


    for query_idx,(class_name, image_path) in enumerate(queries):
        print(query_idx)
        full_path = os.path.join(ANNOTATION_FOLDER_PATH, Path(image_path).stem + '.xml')
        tree = ET.parse(full_path)
        root = tree.getroot()

        objects = root.findall('object')
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        for object in objects:
            try:
                identified_class = object.find('name').text
                if(class_name != identified_class):
                    continue
                bnd_box = object.find('bndbox')
                xmin = int(bnd_box.find('xmin').text)
                ymin = int(bnd_box.find('ymin').text)
                xmax = int(bnd_box.find('xmax').text)
                ymax = int(bnd_box.find('ymax').text)
                break
            except:
                raise Exception('Could not identify the bbox')


        easy_idx_list = []

        # If this query is an easy one.
        if(query_idx % 2 == 0):
            current_easy = dataset[class_name]['easy']
            for relevant_image in current_easy:
                if(relevant_image  in qimlist):
                    continue
                else:
                    idx = imlist.index(relevant_image)
                    if(idx not in easy_idx_list):
                        easy_idx_list.append(idx)                      

            item = {
                'bbx' : [xmin,ymin,xmax,ymax],
                'easy' : easy_idx_list,
                'hard' : [],
                'junk' : [],
                'class' : class_name,
            }

            gnd.append(item)

        else:
            relevant_classes = related_to_categories[image_path]

            all_with_at_least_one = []
            for relevant_class in relevant_classes:
                current_related = []
                current_easy = dataset[relevant_class]['easy']
                for relevant_image in current_easy:
                    if(relevant_image in qimlist):
                        continue
                    else:
                        idx = imlist.index(relevant_image)
                        current_related.append(idx)
                all_with_at_least_one.append(current_related)

            related_gt = list(reduce(set.intersection, [set(item) for item in all_with_at_least_one]))


            item = {
                'bbx' : None,
                'easy' : sorted(related_gt),
                'hard' : [],
                'junk' : [],
                'class' : ' '.join(relevant_classes),
            }

            gnd.append(item)

    return {
        'qimlist' : np.array(qimlist),
        'imlist'  : np.array(imlist),
        'gnd'     : np.array(gnd)
    }    

            


def create_dataset():
    avoid_cfg = None
    
    with open('gnd_pascalvoc_700_medium.pkl','rb') as handle:
       avoid_cfg =  pickle.load(handle)

    list_annotations(ANNOTATION_FOLDER_PATH)
    cfg = create_cnn_cfg(avoid_queries = avoid_cfg['qimlist'])


    with open('gnd_pascalvoc_700_medium_train.pkl','wb') as train_handle:
        pickle.dump(cfg, train_handle, protocol=pickle.HIGHEST_PROTOCOL)
 


create_dataset()