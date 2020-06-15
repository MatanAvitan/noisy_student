FROM python:3.6.10-slim

# Install some basic utilities
RUN apt-get update && \
    apt-get install -y \
    curl \
    sudo \
    ca-certificates \
    git \
    unzip \
    wget \
    time

# specify workdir
WORKDIR /noisy_student

# get data
RUN if  [ ! -d "/noisy_student/src/data-mscoco" ]; then \
        mkdir /noisy_student/src && mkdir /noisy_student/src/data-mscoco && cd /noisy_student/src/data-mscoco && \
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

# create new and original annotations directories, copy files
RUN mkdir /noisy_student/src/data-mscoco/annotations/new
RUN mkdir /noisy_student/src/data-mscoco/annotations/original
RUN mv /noisy_student/src/data-mscoco/annotations/person_keypoints_train2017.json /noisy_student/src/data-mscoco/annotations/original
RUN mv /noisy_student/src/data-mscoco/annotations/person_keypoints_val2017.json /noisy_student/src/data-mscoco/annotations/original

# make output and eval dirs
RUN mkdir /noisy_student/src/outputs
RUN mkdir /noisy_student/src/eval

# Set Environment Variables
ENV ANNOTATIONS_DIR="/noisy_student/src/data-mscoco/annotations"
ENV NEW_ANNOTATIONS_DIR="new"
ENV ORIGINAL_ANNOTATIONS_DIR="original"
ENV ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
ENV ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
ENV OUTPUT_DIR="/noisy_student/src/outputs"
ENV EVAL_DIR="/noisy_student/eval"
ENV COCOSPLIT_PATH="/noisy_student/src/cocosplit.py"

# mkdir data_createor
RUN mkdir /noisy_student/data_creator

# copy requirements
COPY data_requirements.txt /noisy_student/data_creator

# install requirements
RUN pip install -r /noisy_student/data_creator/data_requirements.txt

# copy data splitter
COPY data_splitter.py /noisy_student/src

# copy cocosplit
COPY cocosplit.py /noisy_student/src

# copy init
COPY __init__.py /noisy_student/src

# copy consts
COPY consts.py /noisy_student/src

# split data annotations
RUN python /noisy_student/src/data_splitter.py

# mkdir OUTPUT_DIR
RUN mkdir $OUTPUT_DIR

# remove unecessary files from /noisy_student/src
RUN rm /noisy_student/src/consts.py

RUN rm /noisy_student/src/data_splitter.py

RUN rm /noisy_student/src/cocosplit.py

RUN rm /noisy_student/src/__init__.py
