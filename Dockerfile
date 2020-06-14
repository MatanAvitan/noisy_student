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

# copy requirements
COPY requirements.txt /noisy_student

# install requirements
RUN pip install -r requirements.txt

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

# copy noisy_student src dir
COPY src /noisy_student/src

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
ENV OUTPUT_DIR="/noisy_student/outputs"
ENV EVAL_DIR="/noisy_student/eval"
ENV COCOSPLIT_PATH="/noisy_student/src/cocosplit.py"

# split data annotations
RUN python /noisy_student/src/data_splitter.py

# create openpifpaf directory
RUN cd /noisy_student/src && git clone --single-branch --branch noisy-student https://github.com/atalyaalon/openpifpaf.git

# run noisy student
CMD python /noisy_student/src/noisy_student.py
