# VisualFashionFeaturePrediction
Extraction of fashion product features based on images from the apparel industry

<img href="MISC/header.png">

## Introduction
This repository implements my solution of the <a href="https://www.kaggle.com/c/imaterialist-challenge-fashion-2018">Kaggle iMaterialist Challenge (Fashion) at FCVC5 </a>. The competition's goal was to predict features of products from the apparel industry based on images of the products. The products are selected from a variety of domains e.g. shoes, jackets, necklaces and many more and the target features contain information about the product's category, material, appearance and more. For more information please refer to the competition website or have a look at the `notebooks/iMaterialist_Overview.ipynb` file in which I present a short overview over the dataset.

My personal goal for this competition was to solidify my pytorch skills and to familiarize myself with the <a href="https://github.com/PyTorchLightning/pytorch-lightning">pytorch lightning</a> package, an open-source Python library that provides a high-level interface for PyTorch. For this reason, I solely focused on the training of one model (Resnet34) and did not put much effort into stacking several models together as suggested by the <a href="https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944">winner of the Kaggle competition<a/>.

In addition to the model training pipeline, this repository provides a completely trained model for feature prediction <a href="https://drive.google.com/drive/folders/1EdsssrHV3g1cCNSLd2zke3qwBdNVCOjj?usp=sharing">here</a> and jupyter notebooks for easily scoring new product images and finding the most similar products in a reference dataset, given a query image (see `notebooks` folder). 

The contents of this repository can be used for many applications. An incomplete list of possible topics:

- Analysis of trending attributes
- Competitor Portfolio Analysis
- Intelligent Product Search
- Product Replacements
- Smart Product Recommendation Systems

## Results
#### Feature Prediction
For more information please refer to `notebooks/feature_prediction.ipynb`. If you want to score a pretrained model on own data, please follow the QuickStart instructions.

#### Product Matching
Predicting the features of a product based on it's image allows for matching similar products solely based on their visual appearence. For more information please refer to the jupyter notebook `notebooks/product_matching.ipynb`.

## QuickStart
The following steps will enable you to use a pretrained model to predict the features of a fashion product. You can either use example data provided in this repository or test the model on your own images.

- clone the repository
- download the pretrained model weights from <a href="https://drive.google.com/drive/folders/1EdsssrHV3g1cCNSLd2zke3qwBdNVCOjj?usp=sharing">here</a> (not included in repo due to quota constraints)
- open the jupyter notebook `notebooks/score_model.ipynb`
- follow the notebook instructions

