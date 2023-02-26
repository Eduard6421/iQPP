# iQPP


This readme contains information about the ground truth file structures, instructions for downloading the datasets and preparing it for the retrieval methods.
The dataset follows format of the [Revisted versions of ROxford5K and RParis6k](http://cmp.felk.cvut.cz/revisitop/).

## Schema Format
```
{
    'gnd' : Array<{
        'bbx': [x_min,y_min,x_max,y_max]  | None,
        'easy': Array<number>,
        'medium': Array<number> | None,
        'hard': Array<number> | None
        'junk' : Array<number> | None

    }>,
    'imlist': Array<string>,
    'qimlist': Array<string>
}
```

## Description


1. "qimlist" contains the name of the images that are used for the CBIR. 
2. "imlist" contains the name of the images that can be found in the image database.
3. "gnd" contains:
   1. both bounding boxes for the images (some queries however do not make use of bounding boxes).
   2. three difficulty tracks (easy, medium, hard) which is populated with the index of the images in "imlist", that are positive matches for the query. The tracks easy and medium are not populated for the case of PASCAL VOC and Caltech-101.
   3. "junk" represents images that in certain queries Radenovic et al. [TPAMI 2019] propose to be excluded from the database. However, in our approach, we mark them as negative rather than changing the number of images in the database for each query.
