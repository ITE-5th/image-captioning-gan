#!/usr/bin/env bash

mkdir -p ../data ../data/annotations ../data/train
# download fasttext and convert to gensim
wget --directory-prefix=../data "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip"
unzip ../data/wiki.en.zip -d ../data
python ../misc/fasttext_to_gensim.py

# download coco 2017
wget --directory-prefix=../data "http://images.cocodataset.org/zips/train2017.zip"
unzip ../data/train2017.zip -d ../data/train

# download coco annotation
wget --directory-prefix=../data "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip ../data/annotations_trainval2017.zip -d ../data/annotations

python ../dataset/coco_dataset.py


