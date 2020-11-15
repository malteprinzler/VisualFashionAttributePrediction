# VisualFashionFeaturePrediction
Extraction of fashion product features based on images from the apparel industry

<img href="MISC/header.png">

## Introduction
This repository implements my solution of the <a href="https://www.kaggle.com/c/imaterialist-challenge-fashion-2018">Kaggle iMaterialist Challenge (Fashion) at FCVC5 </a>. The competition's goal was to predict features of products from the apparel industry based on images of the products. The products are selected from a variety of domains e.g. shoes, jackets, necklesses and many more and the target features contained information about the product's category, material, appearence and more. For more information please refer to the competition website or have a look at the `notebooks/iMaterialist_Overview.ipynb` file in which I present a short overview over the dataset.

My personal goal for this competition was to have project to familiarize myself with the <a href="https://github.com/PyTorchLightning/pytorch-lightning">pytorch lightning</a> package, an open-source Python library that provides a high-level interface for PyTorch. For this reason, I focussed only on the training of one model (Resnet34) and did not put much efford into stacking several models together as suggested by the <a href="https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944">winner of the Kaggle competition<a/>.

