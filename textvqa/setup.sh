#!/usr/bin/sh
cd ../textvqa_eval
pwd
# Images
wget --content-disposition https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip

# Training
wget --content-disposition https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget --content-disposition https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json

# Val
wget --content-disposition https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget --content-disposition https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json