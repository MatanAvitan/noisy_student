import os
import logging
from student import Student
from teacher import Teacher
from consts import NUM_TRAIN_EPOCHS, ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, TRAIN_IMAGE_DIR, STUDENT_TEACHER_LOOP, ORIGINAL_VAL_ANNOTATION_FILE, ORIGINAL_ANNOTATIONS_DIR, VAL_IMAGE_DIR, OPENPIFPAF_PATH

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    initial_model_idx = 0
    teacher = Teacher(model_type='openpifpaf',
                      model_idx=initial_model_idx,
                      num_train_epochs=NUM_TRAIN_EPOCHS,
                      train_image_dir=TRAIN_IMAGE_DIR,
                      train_annotations=os.path.join(ANNOTATIONS_DIR,
                                                     NEW_ANNOTATIONS_DIR,
                                                     'annotations_file_model_idx_{model_idx}'.format(model_idx=initial_model_idx)),
                      val_image_dir=VAL_IMAGE_DIR,
                      val_annotations=os.path.join(ANNOTATIONS_DIR,
                                                   ORIGINAL_ANNOTATIONS_DIR,
                                                   ORIGINAL_VAL_ANNOTATION_FILE),
                      next_gen_annotations=os.path.join(ANNOTATIONS_DIR,
                                                        NEW_ANNOTATIONS_DIR,
                                                        'annotations_file_model_idx_{model_idx}'.format(model_idx=initial_model_idx+1)))

    logging.info('********************************************************************')
    logging.info('*************************   Model No {}.    *************************'.format(initial_model_idx))
    logging.info('********************************************************************')
    logging.info('Fitting Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.fit()
    logging.info('Creating Validation Scores to Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.create_val_score()
    logging.info('Creating New data Scores and New Annotations to Model no.{model_idx}'.format(model_idx=initial_model_idx))
    teacher.create_new_data_scores_and_annotations()
    #teacher.save_results()

    for model_idx in range(initial_model_idx+1, STUDENT_TEACHER_LOOP):
        new_student = Student(model_type='openpifpaf',
                              model_idx=model_idx,
                              num_train_epochs=NUM_TRAIN_EPOCHS,
                              train_image_dir=TRAIN_IMAGE_DIR,
                              train_annotations=os.path.join(OPENPIFPAF_PATH,
                                                             'train_annotaions_of_model_no_{model_idx}'.format(model_idx=model_idx)),
                              val_image_dir=VAL_IMAGE_DIR,
                              val_annotations=os.path.join(ANNOTATIONS_DIR,
                                                           ORIGINAL_ANNOTATIONS_DIR,
                                                           ORIGINAL_VAL_ANNOTATION_FILE),
                              next_gen_annotations=os.path.join(ANNOTATIONS_DIR,
                                                                NEW_ANNOTATIONS_DIR,
                                                                'annotations_file_model_idx_{model_idx}'.format(model_idx=model_idx+1)))

        logging.info('********************************************************************')
        logging.info('*************************   Model No {}.    *************************'.format(model_idx))
        logging.info('********************************************************************')
        logging.info('Fitting Model no.{model_idx}'.format(model_idx=model_idx))
        new_student.fit()
        teacher = new_student
        logging.info('Creating Validation Scores to Model no.{model_idx}'.format(model_idx=model_idx))
        teacher.create_val_score()
        if model_idx < STUDENT_TEACHER_LOOP-1:
            logging.info('Creating New data Scores and New Annotations to Model no.{model_idx}'.format(model_idx=model_idx))
            teacher.create_new_data_scores_and_annotations()
        #teacher.save_results()


if __name__ == '__main__':
    main()
