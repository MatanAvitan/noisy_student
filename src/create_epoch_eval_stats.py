import os
import re
import boto3
import logging


OPENPIFPAF_PATH = "/noisy_student/src/openpifpaf"
ANNOTATIONS_DIR="/noisy_student/src/data-mscoco/annotations"
NEW_ANNOTATIONS_DIR="new"
ORIGINAL_ANNOTATIONS_DIR="original"
ORIGINAL_VAL_ANNOTATION_FILE="person_keypoints_val2017.json"
VAL_IMAGE_DIR="/noisy_student/src/data-mscoco/images/val2017"
AWS_ACCESS_ID = os.getenv("AWS_ACCESS_ID")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")

S3_BUCKET_NAME = 'openpifpaf'
experiment_name = 'full-exp-blur'

EVAL_VAL_COMMAND = """cd {openpifpaf_path} && \
                        python -m openpifpaf.eval_coco \
                            --checkpoint {model_output_file} \
                            --long-edge=641 \
                            --loader-workers 16 \
                            --decoder-workers 16 \
                            --batch-size 16 \
                            --dataset other-val \
                            --dataset-image-dir {dataset_image_dir} \
                            --dataset-annotations {dataset_annotations} \
                            --output {eval_output_file}"""

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_ID,
                    aws_secret_access_key=AWS_ACCESS_KEY)

def create_val_score_and_upload_to_s3(checkpoint, metric='oks'):
    """
    creates val score files for val data
    """
    eval_output_file = 'eval_of_val_dataset_checkpoint_{checkpoint}'.format(checkpoint=checkpoint)
    logging.info('Creating val scores of checkpoint {checkpoint}'.format(checkpoint=checkpoint))
    if metric == 'oks':
        eval_process_return_value = os.system(EVAL_VAL_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                                      model_output_file=checkpoint,
                                                                      dataset_image_dir=VAL_IMAGE_DIR,
                                                                      dataset_annotations=os.path.join(ANNOTATIONS_DIR,
                                                                                                       ORIGINAL_ANNOTATIONS_DIR,
                                                                                                       ORIGINAL_VAL_ANNOTATION_FILE),
                                                                      eval_output_file=eval_output_file))
        if eval_process_return_value != 0:
            raise ValueError('Could not create val score - Eval of val failed')
        logging.info('eval_process_return_value:{return_value}'.format(return_value=eval_process_return_value))

        eval_output_stats_file_name = eval_output_file + '.stats.json'
        eval_output_stats_file_path = os.path.join(OPENPIFPAF_PATH, eval_output_stats_file_name)
        if os.path.exists(eval_output_stats_file_path):
            s3.meta.client.upload_file(eval_output_stats_file_path, S3_BUCKET_NAME, os.path.join(experiment_name, 'epocs_val_stats', eval_output_stats_file_name))
        logging.info('Finished Saving Results of checkpoint {checkpoint}'.format(checkpoint=checkpoint))

def main():
    for root,dirs,files in os.walk(OPENPIFPAF_PATH):
        for file in files:
            r1 = re.compile('epoch')
            if r1.search(file) and int(file[-3:]) % 10 == 0:
                filepath = os.path.join(root, file)
                create_val_score_and_upload_to_s3(filepath)


if __name__ == '__main__':
    main()
