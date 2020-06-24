import os

TRAIN_SPLIT_PERCENTAGES = [0.25, 0.333334, 0.5]
STUDENT_TEACHER_LOOP = 3
NUM_TRAIN_EPOCHS = 3
ANNOTATIONS_SCORE_THRESH = 0.7
COCOSPLIT_PATH = os.getenv("COCOSPLIT_PATH")
ANNOTATIONS_DIR = os.getenv('ANNOTATIONS_DIR')
NEW_ANNOTATIONS_DIR = os.getenv('NEW_ANNOTATIONS_DIR')
ORIGINAL_ANNOTATIONS_DIR = os.getenv('ORIGINAL_ANNOTATIONS_DIR')
ORIGINAL_TRAIN_ANNOTATION_FILE = os.getenv('ORIGINAL_TRAIN_ANNOTATION_FILE')
ORIGINAL_VAL_ANNOTATION_FILE = os.getenv('ORIGINAL_VAL_ANNOTATION_FILE')
TRAIN_IMAGE_DIR = os.getenv('TRAIN_IMAGE_DIR')
VAL_IMAGE_DIR = os.getenv('VAL_IMAGE_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
EVAL_DIR = os.getenv('EVAL_DIR')
OPENPIFPAF_PATH = os.getenv('OPENPIFPAF_PATH')

TRAIN_COMMAND = """cd {openpifpaf_path} && \
                   python -m openpifpaf.train \
                       --lr=0.1 \
                       --momentum=0.9 \
                       --epochs={num_train_epochs} \
                       --lr-warm-up-epochs=1 \
                       --lr-decay 120 \
                       --lr-decay-epochs=20 \
                       --lr-decay-factor=0.1 \
                       --batch-size=32 \
                       --square-edge=385 \
                       --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2 \
                       --auto-tune-mtl \
                       --weight-decay=1e-5 \
                       --update-batchnorm-runningstatistics \
                       --ema=0.01 \
                       --basenet=shufflenetv2k16w \
                       --headnets cif caf caf25 \
                       --coco-train-image-dir {train_image_dir} \
                       --cocokp-train-annotations {train_annotations} \
                       --coco-val-image-dir {val_image_dir} \
                       --cocokp-val-annotations {val_annotations} \
                       --output={model_output_file}"""

EVAL_OTHER_COMMAND = """cd {openpifpaf_path} && \
                        python -m openpifpaf.eval_coco \
                            --checkpoint {model_output_file} \
                            -n 500 \
                            --long-edge=641 \
                            --write-predictions \
                            --dataset other \
                            --dataset-image-dir {dataset_image_dir} \
                            --dataset-annotations {dataset_annotations} \
                            --output {eval_output_file}"""
