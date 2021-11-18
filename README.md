---
title: 'VRDL HW2 0716228'
disqus: hackmd
---

VRDL HW2 0716228
===



## Table of Contents

[TOC]

## How to generate answer.json?

1. Download inference.ipynb
2. open inference.ipynb in  google colab 
3. run all cell and download answer.json in mmdection file 

## How to train the model
1.  git clone this project
2.  Download train dataset  from [here](https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc) and unzip it 
3.  `pip install -r requirements.txt`
4.  `python train.py --epoch 7 --batch-size 6`
5.  `model.pt` is your model

## Pretrained model link
[link](https://drive.google.com/file/d/1BjEphetc2ymWuciXDHNZJKPAmIAbwIRl/view?usp=sharing)

#### only `inference.ipynb` and `train.py` were wirtten by me, so others don't follow PEP8

###### tags: `VRDL` `Faster Rcnn`