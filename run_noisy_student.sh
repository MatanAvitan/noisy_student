#!/bin/bash

set -e
# define environment vars

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
export OPENPIFPAF_PATH="src/openpifpaf"
export TRAIN_IMAGE_DIR="src/data-mscoco/images/train2017"
export VAL_IMAGE_DIR="src/data-mscoco/images/val2017"

# use created virtualenv
source ~/.venvs/noisy_student_36/bin/activate

# run noisy student
python src/noisy_student.py
