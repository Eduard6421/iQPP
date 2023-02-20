# iQPP


This readme contains information about the ground truth file structures, instructions for downloading the datasets and preparing it for the retrieval methods.
The dataset follows format of the [Revisted versions of ROxford5K and RParis6k](http://cmp.felk.cvut.cz/revisitop/).

The schema format:

{
    'gnd' : {
        'bbx': [number]  | None,
        'easy': [number],
        'medium': [number] | None,
        'hard': [number] | None
        'junk' : [number] | None

    }[],
    'imlist': string[],
    'qimlist': string[]
}

qimlist contains the name of the images that are used for the CBIR.
imlist constiins the name of the images that can be found in the image database.
gnd contains both bounding boxes for the images (some queries however do not make use of bounding boxes). the bounding boxes if present respect the following format: [x_min,y_min,x_max,y_max].
gnd also contains three difficulty tracks (easy, medium, hard). However these might not always be populated (especially in the case of PASCAL VOC 2012 and Caltech101).  junk represents images that in the  difficulty tracks proposed by Radenovic et al. were excluded. In our approach we mark them as negative rather than having different database images for each search.