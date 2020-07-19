import os
import logging
from model import Model
from consts import TRAIN_COMMAND_STUDENT, BLUR_MAX_SIGMA
from data_consts import OPENPIFPAF_PATH


class Student(Model):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, original_train_annotations, val_image_dir, val_annotations, next_gen_annotations, full_data_model=False):
        super().__init__(model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, original_train_annotations, val_image_dir, val_annotations, next_gen_annotations, full_data_model)

    def fit(self):
        train_images_count = self.get_images_count_in_train_annotations_file()
        train_process_return_value = os.system(TRAIN_COMMAND_STUDENT.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                                            num_train_epochs=self._num_train_epochs,
                                                                            train_image_dir=self._train_image_dir,
                                                                            blur_max_sigma=BLUR_MAX_SIGMA,
                                                                            train_annotations=self._train_annotations,
                                                                            val_image_dir=self._val_image_dir,
                                                                            val_annotations=self._val_annotations,
                                                                            model_output_file=self._model_output_file))
        if train_process_return_value != 0:
            raise ValueError('Training Student failed')
        logging.info('train process of Model no. {model_idx} with {train_images_count} train images return value:{return_value}'.format(model_idx=self._model_idx,
                                                                                                                                        train_images_count=train_images_count,
                                                                                                                                        return_value=train_process_return_value))
