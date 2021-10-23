#!/bin/bash

apt-get update && apt-get install git wget fontconfig python3-pip
mkdir -p ~/.fonts
wget -O ~/.fonts/unifont-13.0.05.ttf http://unifoundry.com/pub/unifont/unifont-13.0.05/font-builds/unifont-13.0.05.ttf
fc-cache -fv
git clone https://github.com/LenardJSchnakenbeck/deobfuscator.git
cd deobfuscator
mkdir -p data/{images,imagesrotated}
mkdir BrandNewModel

pip3 install --upgrade pip &&
pip3 install pillow tensorflow==2.3 pandas tensorboard pandas numpy Wand scikit-learn keras

export CUDA_VISIBLE_DEVICES=0


