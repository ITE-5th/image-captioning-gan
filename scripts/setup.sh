#!/usr/bin/env bash

cd ~; git clone https://github.com/cocodataset/cocoapi.git; cd ~/cocoapi/PythonAPI; make install
cd ~; git clone https://github.com/dmarnerides/pydlt.git; ~/pydlt; pip install . --extra-index-url --trusted-host --upgrade
pip install --no-cache-dir -I pillow

mkdir -p ../data ../data/annotations
# download fasttext and convert to gensim
wget --directory-prefix=../data "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip"
unzip ../data/wiki.en.zip -d ../data
python ../misc/fasttext_to_gensim.py
rm ../data/wiki.en.zip

# download coco 2017
wget --directory-prefix=../data "http://images.cocodataset.org/zips/train2017.zip"
unzip ../data/train2017.zip -d ../data/
mv ../data/train2017 ../data/train
rm ../data/train2017.zip

# download coco annotation
wget --directory-prefix=../data "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip ../data/annotations_trainval2017.zip -d ../data/
mv ../data/annotations_trainval2017 ../data/annotations
rm ../data/annotations_trainval2017.zip

#python ../dataset/coco_dataset.py
