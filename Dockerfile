FROM nvidia/cuda

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

RUN sudo apt install -y --reinstall software-properties-common

RUN sudo add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y \
    python3-pip python3.6 \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3.6 python \
    && ln -s /usr/bin/pip3 pip \
  && pip install --upgrade pip

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

# Set Environment Variables
ENV ANNOTATIONS_DIR="/noisy_student/src/data-mscoco/annotations"
ENV NEW_ANNOTATIONS_DIR="new"
ENV ORIGINAL_ANNOTATIONS_DIR="original"
ENV ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
ENV ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
ENV OUTPUT_DIR="/noisy_student/src/outputs"
ENV EVAL_DIR="/noisy_student/src/eval"
ENV COCOSPLIT_PATH="/noisy_student/src/cocosplit.py"

# mkdir data_createor
RUN mkdir /noisy_student/data_creator

# copy requirements
COPY data_requirements.txt /noisy_student/data_creator

# install requirements
RUN pip install -r /noisy_student/data_creator/data_requirements.txt

# copy data splitter
COPY src/data_splitter.py /noisy_student/src

# copy cocosplit
COPY src/cocosplit.py /noisy_student/src

# copy init
COPY src/__init__.py /noisy_student/src

# copy consts
COPY src/data_consts.py /noisy_student/src/

# split data annotations
RUN python /noisy_student/src/data_splitter.py

# mkdir OUTPUT_DIR
RUN mkdir $OUTPUT_DIR

# mkdir EVAL_DIR
RUN mkdir $EVAL_DIR

# copy requirements
COPY requirements.txt /noisy_student

# install requirements
RUN pip install -r requirements.txt

# create src dir and copy noisy_student src dir
RUN mkdir /noisy_student/src_code
COPY src /noisy_student/src_code

# mv src_code dir content into src dir
RUN mv /noisy_student/src_code/* /noisy_student/src/

# delete src_code dir
RUN rmdir /noisy_student/src_code/

# Set Additional Environment Variables
ENV OPENPIFPAF_PATH="/noisy_student/src/openpifpaf"
ENV TRAIN_IMAGE_DIR="/noisy_student/src/data-mscoco/images/train2017"
ENV VAL_IMAGE_DIR="/noisy_student/src/data-mscoco/images/val2017"
ENV AWS_CREDENTIALS_FILE_PATH=$AWS_CREDENTIALS_FILE_PATH

# create openpifpaf directory
RUN cd /noisy_student/src && git clone --single-branch --branch noisy-student https://github.com/atalyaalon/openpifpaf.git

# Add AWS Credentials file
RUN mkdir ~/.aws
COPY $AWS_CREDENTIALS_FILE_PATH ~/.aws/credentials

# run noisy student
CMD python /noisy_student/src/noisy_student.py
