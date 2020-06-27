import os

MOCK_RUN = os.getenv("MOCK_RUN")
MOCK_ONE_MODEL = os.getenv("MOCK_ONE_MODEL")
if MOCK_RUN == 'TRUE':
    NUM_TRAIN_EPOCHS = 3
    ANNOTATIONS_SCORE_THRESH = 0
else:
    NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS"))
    ANNOTATIONS_SCORE_THRESH = float(os.getenv("ANNOTATIONS_SCORE_THRESH"))

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
AWS_ACCESS_ID = os.getenv("AWS_ACCESS_ID")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
CREATE_IMAGES = os.getenv("CREATE_IMAGES")

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
                       --loader-workers 24 \
                       --coco-train-image-dir {train_image_dir} \
                       --cocokp-train-annotations {train_annotations} \
                       --coco-val-image-dir {val_image_dir} \
                       --cocokp-val-annotations {val_annotations} \
                       --output={model_output_file}"""

EVAL_COMMAND = """cd {openpifpaf_path} && \
                        python -m openpifpaf.eval_coco \
                            --checkpoint {model_output_file} \
                            --long-edge=641 \
                            --write-predictions \
                            --loader-workers 16 \
                            --decoder-workers 16 \
                            --batch-size 16 \
                            --dataset other \
                            --dataset-image-dir {dataset_image_dir} \
                            --dataset-annotations {dataset_annotations} \
                            --output {eval_output_file}"""

PREDICT_COMMAND = """cd {openpifpaf_path} && \
                        python -m openpifpaf.predict \
                            {images} \
                            --checkpoint {checkpoint} \
                            --image-output {image_output_dir}"""
