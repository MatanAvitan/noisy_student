import os

TRAIN_SPLIT_PERCENTAGES = [0.25, 0.333334, 0.5]
STUDENT_TEACHER_LOOP = 3
NUM_TRAIN_EPOCHS = 5
ANNOTATIONS_DIR = os.getenv('ANNOTATIONS_DIR')
NEW_ANNOTATIONS_DIR = os.getenv('NEW_ANNOTATIONS_DIR')
ORIGINAL_ANNOTATIONS_DIR = os.getenv('ORIGINAL_ANNOTATIONS_DIR')
ORIGINAL_TRAIN_ANNOTATION_FILE = os.getenv('ORIGINAL_TRAIN_ANNOTATION_FILE')
TRAIN_IMAGE_DIR = os.getenv('TRAIN_IMAGE_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
EVAL_DIR = os.getenv('OUTPUT_DIR')

TRAIN_COMMAND = """python -m openpifpaf.train \
                      --lr=0.05 \
                      --momentum=0.9 \
                      --epochs={num_train_epochs} \
                      --lr-warm-up-epochs=1 \
                      --lr-decay 220 \
                      --lr-decay-epochs=30 \
                      --lr-decay-factor=0.01 \
                      --batch-size=32 \
                      --square-edge=385 \
                      --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2 \
                      --auto-tune-mtl \
                      --weight-decay=1e-5 \
                      --update-batchnorm-runningstatistics \
                      --ema=0.01 \
                      --train-image-dir {train_image_dir} \
                      --train-annotations {train_annotations} \
                      --output={model_output_file}"""

EVAL_VAL_COMMAND = """python -m openpifpaf.eval_coco \
                  --checkpoint {model_output_file} \
                  -n 500 \
                  --long-edge=641 \
                  --write-predictions \
                  --dataset val \
                  --output {eval_output_file}"""

EVAL_OTHER_COMMAND = """python -m openpifpaf.eval_coco \
                  --checkpoint {model_output_file} \
                  -n 500 \
                  --long-edge=641 \
                  --write-predictions \
                  --dataset other \
                  --dataset-image-dir {dataset_image_dir} \
                  --dataset-annotations {dataset_annotations} \
                  --output {eval_output_file}"""
