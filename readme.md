# iQPP

We release our code as an open source license however the retrieval methods and the datasets studied each have their own license and should be respected.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

Retrieval methods:
1. [Radenovic et al.](https://github.com/filipradenovic/cnnimageretrieval-pytorch); - MIT License
2. [Revaud et al.](https://github.com/naver/deep-image-retrieval); BSD-3 Clause license

Datasets:
1. (ROxford5k (Download))[http://cmp.felk.cvut.cz/revisitop/] - [Flickr terms of use](https://www.flickr.com/help/terms) and [Dataset Terms of Access](https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html)
2. (RParis6k (Download))[http://cmp.felk.cvut.cz/revisitop/] - [Flickr terms of use](https://www.flickr.com/help/terms) and [Dataset Terms of Access](https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html)
3. (PASCAL VOC 2012 (Download))[http://host.robots.ox.ac.uk/pascal/VOC/] - [Flickr terms of use](https://www.flickr.com/help/terms)
4. (Caltech-101)[https://data.caltech.edu/records/mzrjq-6wc02] - [Caltech Data terms of use](https://library.caltech.edu/search/caltechdata#terms)

---

## 📝 Table of Contents

- [iQPP](#iqpp)
  - [📝 Table of Contents](#-table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [Installing Prerequisites](#installing-prerequisites)
  - [Usage ](#usage-)
  - [⛏️ Developed With ](#️-developed-with-)
  - [🎉 Acknowledgements ](#-acknowledgements-)

## About <a name = "about"></a>

This repository holds the source code four our paper "Image Query Performance Prediction". It holds the label generation scripts, modifications to the retrieval models and implementations of the studied models.

## Getting Started <a name = "getting_started"></a>

The instructions will allow you to copy the project and run all the retrieval methods and benchmarked models on you local machine
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites


Prerequisites can be seen in the requirements.txt text file.
However I would like to point out that Revaud et al. retrieval model cannot be run on an newer scikit-learn version. As per the [issue](https://github.com/naver/deep-image-retrieval/issues/27) tuhe authors recommend the version 0.20.2.

### Installing Prerequisites


Run the following command in order to install the prerequisites:

```
  pip install -r requirements.txt
```

## Usage <a name="usage"></a>

There are multiple steps involved in running the benchmark. In order to replicate our results you will have to follow the next steps:
1. Download the datasets from the provided links;
2. Copy all the images in the respective Dataset folder under the "jpg" folder. For example in case of ROxford5K you must copy the images in Datasets/ROxford5k/jpg;
3. Run the retrieval methods
    Note. The methods do not have a unified interface and us such require method specific arguments and changes. All of these will be described in detail.
    
    3.1 Radenovic et al.

        3.1.1 Decide on which dataset you want to run the retrieval model. Available options are: `roxford5k, rparis6k, pascalvoc_700_medium,pascalvoc_700_medium_train,caltech101_700,caltech101_700_train, pascalvoc_700_medium_drift_cnn, pascalvoc_700_no_bbx_drift_cnn, caltech101_700_drift_cnn`.
        The methods marked as drift denote the queries were identified during the adapted query drift technique.
      
        3.1.2 If you wish to save the top-100 results and embeddings of the queries / dataset you must make the following modifications;

            top-100-results: in Retrieval_Methods\Radenovic-et-al\radenovic-et-al\cirtorch\utils\evaluate.py line 144. Decomment and change the name of the file depending on your dataset;
          
            query and db embeddings: in Retrieval_Methods\Radenovic-et-al\radenovic-et-al\cirtorch\examples\test.py lines 256/260. Decomment and change the name of the file depending on your dataset;
          
        3.1.3 Select the metric you want to show (p@100 or ap) by changing which value you want to be printed;

            in Retrieval_Methods\Radenovic-et-al\radenovic-et-al\cirtorch\utils\evaluate.py comment one of the lines 135/137. by default AP is selected.
          
        3.1.4 Run the model with the following instructions:  
            ```
                  python -m cirtorch.examples.test --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' \
                    --datasets 'caltech101_700' \
                    --whitening 'retrieval-SfM-120k' \
                    --multiscale '[1, 1/2**(1/2), 1/2]'
            ```
    3.2 Revaud et al.

        3.2.1 Decide on which dataset you want to run the retrieval model. Available options are : `ROxford5K,ROxford5K_Drift, RParis6K,RParis6K_Drift,PascalVOC_700_Medium,PascalVOC_700_Medium_Train,PascalVOC_700_Medium_Drift,Caltech101_700, Caltech101_700_Train,Caltech101_700_Drift`

        3.2.2 If you wish to save the top-100 results and embeddings of the queries/ database  you mist make the following modifications:

          top-100-results: in Retrieval_Methods\Revaud-et-al\deep-image-retrieval\dirtorch\test_dir.py line 160. Decomment and change the name of the file depending on your dataset.
          query and db embeddings: in Retrieval_Methods\Revaud-et-al\deep-image-retrieval\dirtorch\test_dir.py lines 160/161. Decomment and change the name of the file depending on your dataset.

        3.2.3 Run the model with the following instructions:
          python -m dirtorch.test_dir --dataset 'Caltech101_700' --checkpoint '/notebooks/deep-image-retrieval/Resnet101-AP-GeM.pt'
        ```
4. Run the QPP models 

5. Compute the correlations

## ⛏️ Developed With <a name = "developed_using"></a>
- [Pytorch](https://pytorch.org/) - Deel Learning Library

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

Special thank to all the dataset collectors, adnotators and for the researchers behind the content-based image retrieval methods!
