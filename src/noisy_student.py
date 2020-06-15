from student import Student
from teacher import Teacher
from consts import NUM_TRAIN_EPOCHS, ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, TRAIN_IMAGE_DIR, STUDENT_TEACHER_LOOP


def main():
    initial_model_idx = 0
    teacher = Teacher(model_type='openpifpaf',
                      model_idx=initial_model_idx,
                      num_train_epochs=NUM_TRAIN_EPOCHS,
                      train_image_dir=TRAIN_IMAGE_DIR,
                      train_annotations=os.path.join(ANNOTATIONS_DIR,
                                                     NEW_ANNOTATIONS_DIR,
                                                     'annotations_file_model_idx_{model_idx}'.format(model_idx=initial_model_idx)))

    teacher.fit()
    teacher.get_val_score()
    teacher.create_new_data_scores_and_annotations()
    teacher.save_results()

    for model_idx in range(initial_model_idx+1, STUDENT_TEACHER_LOOP):
        new_student = Student(model_type='openpifpaf',
                              model_idx=model_idx,
                              num_train_epochs=NUM_TRAIN_EPOCHS,
                              train_image_dir=TRAIN_IMAGE_DIR,
                              train_annotations=os.path.join(ANNOTATIONS_DIR,
                                                             NEW_ANNOTATIONS_DIR,
                                                            'annotations_file_model_idx_{model_idx}'.format(model_idx=model_idx)))

                                                             # TODO: change to: train_annotations=teacher._new_train_annotations once create_new_annotations_file(self) is implemented in teacher

        new_student.fit()
        teacher = new_student
        teacher.get_val_score()
        teacher.create_new_data_scores_and_annotations()
        teacher.save_results()


if __name__ == '__main__':
    main()
