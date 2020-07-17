import os
import logging
import boto3
from torch.utils.tensorboard import SummaryWriter
from student import Student
from teacher import Teacher
from consts import (NUM_TRAIN_EPOCHS,
                    S3_BUCKET_NAME,
                    EXPERIMENT_NAME,
                    AWS_ACCESS_ID,
                    AWS_ACCESS_KEY,
                    MOCK_ONE_MODEL,
                    ANNOTATIONS_SCORE_INITIAL_THRESH,
                    ANNOTATION_SCORE_DECREASE,
                    STUDENT_TEACHER_LOOP,
                    RUN_FULL_MODEL)
from data_consts import (ANNOTATIONS_DIR,
                         NEW_ANNOTATIONS_DIR,
                         TRAIN_IMAGE_DIR,
                         ORIGINAL_ANNOTATIONS_DIR,
                         ORIGINAL_VAL_ANNOTATION_FILE,
                         VAL_IMAGE_DIR,
                         OPENPIFPAF_PATH,
                         NEW_ANNOTATIONS_FILE_PREFIX,
                         MERGED_TRAIN_ANNOTATIONS_FILE_PREFIX,
                         ANNOTATIONS_FILE_FULL_MODEL)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

def create_results_dir_in_s3(experiment_name):
    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_ID,
                      aws_secret_access_key=AWS_ACCESS_KEY)
    bucket_name = S3_BUCKET_NAME
    directory_name = experiment_name + '/'
    s3.put_object(Bucket=bucket_name, Key=directory_name, Body='')

def upload_tb_logs_to_s3(experiment_name):
    logging.info('Uploading TB Logs Dir to S3 for Experiment {experiment_name}'.format(experiment_name=experiment_name))
    tb_logs_path = os.path.join(OPENPIFPAF_PATH, 'tb_logs')
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_ID,
                        aws_secret_access_key=AWS_ACCESS_KEY)
    for root,dirs,files in os.walk(tb_logs_path):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.exists(filepath):
                logging.info('Uploading to Bucket {}, Experiment {}, Filename {}'.format(S3_BUCKET_NAME, experiment_name, file))
                s3.meta.client.upload_file(filepath, S3_BUCKET_NAME, os.path.join(experiment_name, 'tb_logs', file))
                logging.info('Finished Saving TB Logs File {filename} in S3'.format(filename=file))

def create_full_data_model_for_comparison(model_idx):
    full_data_model = Teacher(model_type='openpifpaf',
                              model_idx=model_idx,
                              num_train_epochs=NUM_TRAIN_EPOCHS,
                              train_image_dir=TRAIN_IMAGE_DIR,
                              train_annotations=os.path.join(ANNOTATIONS_DIR,
                                                             NEW_ANNOTATIONS_DIR,
                                                             ANNOTATIONS_FILE_FULL_MODEL),
                              original_train_annotations=None,
                              val_image_dir=VAL_IMAGE_DIR,
                              val_annotations=os.path.join(ANNOTATIONS_DIR,
                                                           ORIGINAL_ANNOTATIONS_DIR,
                                                           ORIGINAL_VAL_ANNOTATION_FILE),
                              next_gen_annotations=None,
                              full_data_model=True)
    return full_data_model

def main():
    create_results_dir_in_s3(experiment_name=EXPERIMENT_NAME)
    # variables: initial_model_idx, curr_thresh
    initial_model_idx = 0
    curr_thresh = ANNOTATIONS_SCORE_INITIAL_THRESH
    original_train_annotations = os.path.join(ANNOTATIONS_DIR,
                                              NEW_ANNOTATIONS_DIR,
                                              '{prefix}_{model_idx}'.format(prefix=NEW_ANNOTATIONS_FILE_PREFIX,
                                                                            model_idx=initial_model_idx))
    next_gen_annotations = os.path.join(ANNOTATIONS_DIR,
                                        NEW_ANNOTATIONS_DIR,
                                        '{prefix}_{model_idx}'.format(prefix=NEW_ANNOTATIONS_FILE_PREFIX,
                                                                      model_idx=initial_model_idx+1))
    teacher = Teacher(model_type='openpifpaf',
                      model_idx=initial_model_idx,
                      num_train_epochs=NUM_TRAIN_EPOCHS,
                      train_image_dir=TRAIN_IMAGE_DIR,
                      train_annotations=original_train_annotations,
                      original_train_annotations=original_train_annotations,
                      val_image_dir=VAL_IMAGE_DIR,
                      val_annotations=os.path.join(ANNOTATIONS_DIR,
                                                   ORIGINAL_ANNOTATIONS_DIR,
                                                   ORIGINAL_VAL_ANNOTATION_FILE),
                      next_gen_annotations=next_gen_annotations)

    logging.info('********************************************************************')
    logging.info('*************************   Model No {model_idx}.    *************************'.format(model_idx=initial_model_idx))
    logging.info('********************************************************************')
    logging.info('Fitting Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.fit()
    logging.info('Creating Validation Scores to Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.create_val_score()
    logging.info('Creating New data Scores and New Annotations to Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.create_new_data_scores_and_annotations(thresh=curr_thresh)
    teacher.save_annotations(experiment_name=EXPERIMENT_NAME)
    curr_thresh = round(max(0, curr_thresh - ANNOTATION_SCORE_DECREASE), 3)
    teacher.save_results(experiment_name=EXPERIMENT_NAME)
    teacher.save_logs(experiment_name=EXPERIMENT_NAME)
    teacher.save_model(experiment_name=EXPERIMENT_NAME)
    if MOCK_ONE_MODEL == 'TRUE':
        return

    for model_idx in range(initial_model_idx+1, STUDENT_TEACHER_LOOP):
        last_model_in_loop = model_idx == STUDENT_TEACHER_LOOP-1
        if not last_model_in_loop:
            curr_next_gen_annotations = next_gen_annotations
        else:
            curr_next_gen_annotations = None
        new_student = Student(model_type='openpifpaf',
                              model_idx=model_idx,
                              num_train_epochs=NUM_TRAIN_EPOCHS,
                              train_image_dir=TRAIN_IMAGE_DIR,
                              train_annotations=os.path.join(OPENPIFPAF_PATH,
                                                             '{prefix}_{model_idx}'.format(prefix=MERGED_TRAIN_ANNOTATIONS_FILE_PREFIX,
                                                                                           model_idx=model_idx)),
                              original_train_annotations=original_train_annotations,
                              val_image_dir=VAL_IMAGE_DIR,
                              val_annotations=os.path.join(ANNOTATIONS_DIR,
                                                           ORIGINAL_ANNOTATIONS_DIR,
                                                           ORIGINAL_VAL_ANNOTATION_FILE),
                              next_gen_annotations=curr_next_gen_annotations)

        logging.info('********************************************************************')
        logging.info('*************************   Model No {model_idx}.    *************************'.format(model_idx=model_idx))
        logging.info('********************************************************************')
        logging.info('Fitting Model no.{model_idx}'.format(model_idx=model_idx))
        new_student.fit()
        teacher = new_student
        logging.info('Creating Validation Scores to Model no.{model_idx}'.format(model_idx=model_idx))
        teacher.create_val_score()
        if not last_model_in_loop:
            logging.info('Creating New data Scores and New Annotations to Model no.{model_idx}'.format(model_idx=model_idx))
            teacher.create_new_data_scores_and_annotations(thresh=curr_thresh)
            teacher.save_annotations(experiment_name=EXPERIMENT_NAME)
            curr_thresh = round(max(0, curr_thresh - ANNOTATION_SCORE_DECREASE), 3)
        teacher.save_results(experiment_name=EXPERIMENT_NAME)
        teacher.save_logs(experiment_name=EXPERIMENT_NAME)
        teacher.save_model(experiment_name=EXPERIMENT_NAME)
    if RUN_FULL_MODEL == 'TRUE':
        full_data_model = create_full_data_model_for_comparison(model_idx+1)
        full_data_model.fit()
        full_data_model.create_val_score()
        full_data_model.save_results(experiment_name=EXPERIMENT_NAME)
        full_data_model.save_logs(experiment_name=EXPERIMENT_NAME)
        full_data_model.save_model(experiment_name=EXPERIMENT_NAME)
    upload_tb_logs_to_s3(experiment_name=EXPERIMENT_NAME)

if __name__ == '__main__':
    main()
