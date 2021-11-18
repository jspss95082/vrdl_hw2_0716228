---
title: 'VRDL HW2 0716228'
disqus: hackmd
---

VRDL HW2 0716228
===


[![hackmd-github-sync-badge](https://hackmd.io/6p7VjUESSYeoBbkQPXsG3Q/badge)](https://hackmd.io/6p7VjUESSYeoBbkQPXsG3Q)
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

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

###### tags: `VRDL` `Faster Rcnn`