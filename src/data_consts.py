import os

TRAIN_SPLIT_PERCENTAGES = [0.333334, 0.5]
STUDENT_TEACHER_LOOP = 3
COCOSPLIT_PATH = os.getenv("COCOSPLIT_PATH", "/noisy_student/src/cocosplit.py")
ANNOTATIONS_DIR = os.getenv('ANNOTATIONS_DIR', "/noisy_student/src/data-mscoco/annotations")
NEW_ANNOTATIONS_DIR = os.getenv('NEW_ANNOTATIONS_DIR', "new")
ORIGINAL_ANNOTATIONS_DIR = os.getenv('ORIGINAL_ANNOTATIONS_DIR', "original")
ORIGINAL_TRAIN_ANNOTATION_FILE = os.getenv('ORIGINAL_TRAIN_ANNOTATION_FILE', "person_keypoints_train2017.json")
ORIGINAL_VAL_ANNOTATION_FILE = os.getenv('ORIGINAL_VAL_ANNOTATION_FILE', "person_keypoints_val2017.json")
TRAIN_IMAGE_DIR = os.getenv('TRAIN_IMAGE_DIR')
VAL_IMAGE_DIR = os.getenv('VAL_IMAGE_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', "/noisy_student/src/outputs")
EVAL_DIR = os.getenv('EVAL_DIR', "/noisy_student/src/eval")
OPENPIFPAF_PATH = os.getenv('OPENPIFPAF_PATH', "/noisy_student/src/openpifpaf")
NEW_ANNOTATIONS_FILE_PREFIX = 'annotations_file_model_idx'
MERGED_TRAIN_ANNOTATIONS_FILE_PREFIX = 'train_annotaions_of_model_no'
