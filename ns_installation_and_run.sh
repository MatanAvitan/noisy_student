#!/bin/sh

apt-get update && \
    apt-get install -y \
    curl \
    sudo \
    ca-certificates \
    git \
    unzip \
    wget \
    time \
    git

sudo apt install -y --reinstall software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa

apt-get install -y \
    python3-pip python3.6 python3.6-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3.6 python \
    && ln -s /usr/bin/pip3 pip \
  && pip install --upgrade pip

# get data
if  [ ! -d "src/data-mscoco" ]; then \
        mkdir src && mkdir src/data-mscoco && cd src/data-mscoco && \
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
        unzip annotations_trainval2017.zip && \
        wget http://images.cocodataset.org/annotations/image_info_test2017.zip && \
        unzip image_info_test2017.zip && \
        mkdir images && cd images && \
        wget http://images.cocodataset.org/zips/val2017.zip && \
        unzip val2017.zip && \
        wget http://images.cocodataset.org/zips/train2017.zip && \
        unzip train2017.zip ; \
fi

mkdir src/data-mscoco/annotations/new
mkdir src/data-mscoco/annotations/original
mv src/data-mscoco/annotations/person_keypoints_train2017.json src/data-mscoco/annotations/original
mv src/data-mscoco/annotations/person_keypoints_val2017.json src/data-mscoco/annotations/original

export ANNOTATIONS_DIR="src/data-mscoco/annotations"
export NEW_ANNOTATIONS_DIR="new"
export ORIGINAL_ANNOTATIONS_DIR="original"
export ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
export ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
export OUTPUT_DIR="src/outputs"
export EVAL_DIR="src/eval"
export COCOSPLIT_PATH="src/cocosplit.py"

# install data requirements
pip install -r src/data_requirements.txt

# split data annotations
python src/data_splitter.py

# mkdir OUTPUT_DIR
mkdir $OUTPUT_DIR

# mkdir EVAL_DIR
mkdir $EVAL_DIR

# install requirements
pip install -r requirements.txt

# Set Additional Environment Variables
export OPENPIFPAF_PATH="src/openpifpaf"
export TRAIN_IMAGE_DIR="src/data-mscoco/images/train2017"
export VAL_IMAGE_DIR="src/data-mscoco/images/val2017"

# create openpifpaf directory
cd src && git clone --single-branch --branch noisy-student https://github.com/atalyaalon/openpifpaf.git

# run noisy student
python src/noisy_student.py
