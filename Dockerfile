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
RUN if  [ ! -d "/noisy_student/data/data-mscoco" ]; then \
        mkdir /noisy_student/data && mkdir /noisy_student/data/data-mscoco && cd /noisy_student/data/data-mscoco && \
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
RUN mkdir /noisy_student/data/data-mscoco/annotations/new
RUN mkdir /noisy_student/data/data-mscoco/annotations/original
RUN mv /noisy_student/data/data-mscoco/annotations/person_keypoints_train2017.json /noisy_student/data/data-mscoco/annotations/original
RUN mv /noisy_student/data/data-mscoco/annotations/person_keypoints_val2017.json /noisy_student/data/data-mscoco/annotations/original

# Set Environment Variables
ENV ANNOTATIONS_DIR="/noisy_student/data/data-mscoco/annotations"
ENV NEW_ANNOTATIONS_DIR="new"
ENV ORIGINAL_ANNOTATIONS_DIR="original"
ENV ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
ENV ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
ENV OUTPUT_DIR="/noisy_student/data/outputs"
ENV EVAL_DIR="/noisy_student/data/eval"
ENV COCOSPLIT_PATH="/noisy_student/data/cocosplit.py"

# mkdir data_createor
RUN mkdir /noisy_student/data_creator

# copy requirements
COPY data_requirements.txt /noisy_student/data_creator

# install requirements
RUN pip install -r /noisy_student/data_creator/data_requirements.txt

# copy data splitter
COPY src/data_splitter.py /noisy_student/data

# copy cocosplit
COPY src/cocosplit.py /noisy_student/data

# copy init
COPY src/__init__.py /noisy_student/data

# copy consts
COPY src/data_consts.py /noisy_student/data/consts.py

# split data annotations
RUN python /noisy_student/data/data_splitter.py

# mkdir OUTPUT_DIR
RUN mkdir $OUTPUT_DIR

# mkdir EVAL_DIR
RUN mkdir $EVAL_DIR

# remove unecessary files from /noisy_student/data
RUN rm /noisy_student/data/consts.py

RUN rm /noisy_student/data/data_splitter.py

RUN rm /noisy_student/data/cocosplit.py

RUN rm /noisy_student/data/__init__.py

# copy requirements
COPY requirements.txt /noisy_student

# install requirements
RUN pip install -r requirements.txt

# create src dir
RUN mkdir /noisy_student/src

# mv data dir content into src dir
RUN mv /noisy_student/data /noisy_student/src

# mkdir src_files and copy src files into src_files dir
RUN mkdir /noisy_student/src_files
COPY src /noisy_student/src_files

# mv src_files into src dir
RUN mv /noisy_student/src_files/* /noisy_student/src

# Set Environment Variables
ENV ANNOTATIONS_DIR="/noisy_student/src/data-mscoco/annotations"
ENV NEW_ANNOTATIONS_DIR="new"
ENV ORIGINAL_ANNOTATIONS_DIR="original"
ENV ORIGINAL_TRAIN_ANNOTATION_FILE="person_keypoints_train2017.json"
ENV ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
ENV OUTPUT_DIR="/noisy_student/src/outputs"
ENV EVAL_DIR="/noisy_student/src/eval"
ENV OPENPIFPAF_PATH="/noisy_student/src/openpifpaf"
ENV TRAIN_IMAGE_DIR="/noisy_student/src/data-mscoco/images/train2017"
ENV VAL_IMAGE_DIR="/noisy_student/src/data-mscoco/images/val2017"

# create openpifpaf directory
RUN cd /noisy_student/src && git clone --single-branch --branch noisy-student https://github.com/atalyaalon/openpifpaf.git

# run noisy student
CMD python /noisy_student/src/noisy_student.py
