import os
import shutil
import scipy.io
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import matplotlib.pyplot as plt 
from IPython.display import display

SRC_FOLDER  = 'caltech101-images'
DEST_FOLDER = 'caltech101'
ANNOTATION_FOLDER = 'Annotations\\Annotations'



#mat = scipy.io.loadmat(file_path)
#img = Image.open(image_path)


#y_min, y_max, x_min , x_max = mat['box_coord'][0]
#print(mat['box_coord'])





def generate_dataset(src_folder, dest_folder, avoid_queries):
    categories = os.listdir(src_folder)

    image_id = 1

    img_categ = {

    }

    dataset_categs = {
    }


    imlist = []

    # Moving items into the same folder and saving the name of the files
    for category in categories:
        subfolder_path = os.path.join(src_folder, category)
        image_names =  os.listdir(subfolder_path)

        for image_name in image_names:
            image_path = os.path.join(subfolder_path, image_name)
            new_image_name = 'image_{}'.format(image_id)
            dest_path = os.path.join(dest_folder,new_image_name ) + '.jpg'
            shutil.copy(image_path, dest_path)
            current_categ_items = dataset_categs.get(category,[])
            current_categ_items.append(new_image_name)
            dataset_categs[category] = current_categ_items
            image_id +=1
            img_categ[new_image_name] = category
            imlist.append(Path(new_image_name).stem)
        

    qimlist = []
    gnd = []
    
    num_queries = 700
    num_classes = len(dataset_categs.keys()) - 1 # removing the clutter clas from queries
    classes_extra = num_queries % num_classes
    num_images    = num_queries // num_classes

    num_images_for_each = np.full((num_classes + 1), num_images)
    num_images_for_each[:(classes_extra+1)] += 1

    for idx, key in enumerate(dataset_categs.keys()):

        if(key == "BACKGROUND_Google"):
            continue

        num_of_images = num_images_for_each[idx]

        cnt = 0
        for i in range(len(dataset_categs[key])):
            if(dataset_categs[key][i] not in avoid_queries):
                qimlist.append(dataset_categs[key][i])
                cnt+=1
            if(cnt == num_of_images):
                break
    
    imlist = [item for item in imlist if item not in qimlist]

    for query_image_name in qimlist:
        query_category =  img_categ[query_image_name]
        related_images = dataset_categs[query_category]

        gt_images = []
        for related_image in related_images:
            if(related_image in imlist):
                index = imlist.index(related_image)
                gt_images.append(index)

        item = {
            'bbx' : None,
            'easy' : gt_images,
            'hard' : [],
            'junk' : [],
            'class' : query_category
        }

        gnd.append(item)

    return {
        'qimlist': qimlist,
        'imlist' : imlist,
        'gnd' : gnd
    }


if __name__ == "__main__":
    

    avoid_cfg = None
    with open('gnd_caltech101_700.pkl','rb') as handle:
        avoid_cfg = pickle.load(handle)
    
    cfg = generate_dataset(src_folder=SRC_FOLDER,dest_folder=DEST_FOLDER,avoid_queries = avoid_cfg['qimlist'])

    with open('gnd_caltech101_700_train.pkl','wb') as handle:
        pickle.dump(cfg, handle, protocol=pickle.HIGHEST_PROTOCOL)