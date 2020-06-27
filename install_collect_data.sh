#!/bin/bash

set -e
sudo apt-get update && \
    sudo apt-get install -y \
    curl \
    ca-certificates \
    git \
    unzip \
    wget \
    time

sleep 2

sudo apt install -y --reinstall software-properties-common

sleep 2

sudo add-apt-repository -y ppa:deadsnakes/ppa

sleep 2

sudo apt-get remove -y python3-apt

sleep 2

sudo apt-get install -y --reinstall python3-apt

sleep 2

sudo apt-get install -y --reinstall python3-software-properties


sleep 2

sudo apt-get install -y \
    python3-pip python3.6 python3.6-dev

sleep 2

# create virtualenv
sudo apt-get -y install virtualenv
virtualenv -p python3 ~/.venvs/noisy_student_36
source ~/.venvs/noisy_student_36/bin/activate

sleep 2

# get data
if  [ ! -d "src/data-mscoco" ]; then \
        mkdir src/data-mscoco && cd src/data-mscoco && \
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
        unzip annotations_trainval2017.zip && \
        wget http://images.cocodataset.org/annotations/image_info_test2017.zip && \
        unzip image_info_test2017.zip && \
        mkdir images && cd images && \
        wget http://images.cocodataset.org/zips/val2017.zip && \
        unzip val2017.zip && \
        wget http://images.cocodataset.org/zips/train2017.zip && \
        unzip train2017.zip && \
        cd ../../.. ; \
fi

sleep 2

sudo mkdir src/data-mscoco/annotations/new
sudo chmod -R o+rwx src/data-mscoco/annotations/new
sudo mkdir src/data-mscoco/annotations/original
sudo chmod -R o+rwx src/data-mscoco/annotations/original
sudo mv src/data-mscoco/annotations/person_keypoints_train2017.json src/data-mscoco/annotations/original
sudo mv src/data-mscoco/annotations/person_keypoints_val2017.json src/data-mscoco/annotations/original

export ANNOTATIONS_DIR="src/data-mscoco/annotations"
export NEW_ANNOTATIONS_DIR="new"
export ORIGINAL_ANNOTATIONS_DIR="original"
export ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
export ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
export OUTPUT_DIR="src/outputs"
export EVAL_DIR="src/eval"
export COCOSPLIT_PATH="src/cocosplit.py"
export OPENPIFPAF_PATH="src/openpifpaf"
export TRAIN_IMAGE_DIR="src/data-mscoco/images/train2017"
export VAL_IMAGE_DIR="src/data-mscoco/images/val2017"

sleep 2

# install data requirements
pip install -r data_requirements.txt

sleep 2

# split data annotations
python src/data_splitter.py

sleep 2

# mkdir OUTPUT_DIR
sudo mkdir $OUTPUT_DIR

# mkdir EVAL_DIR
sudo mkdir $EVAL_DIR

sleep 2

# install requirements
pip install -r requirements.txt

# Set Additional Environment Variables
export OPENPIFPAF_PATH="src/openpifpaf"
export TRAIN_IMAGE_DIR="src/data-mscoco/images/train2017"
export VAL_IMAGE_DIR="src/data-mscoco/images/val2017"

sleep 2

# create openpifpaf directory
cd src && git clone --single-branch --branch noisy-student https://github.com/atalyaalon/openpifpaf.git && cd ..
