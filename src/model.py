import os
import logging
from consts import TRAIN_COMMAND, EVAL_OTHER_COMMAND, OPENPIFPAF_PATH

class Model(object):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, val_image_dir, val_annotations, next_gen_annotations):
        self._model_type = model_type
        self._model_idx = model_idx
        self._model_output_file = 'model_type_{model_type}_model_no_{model_idx}'.format(model_idx=model_idx,
                                                                                                     model_type=model_type)
        self._eval_output_file = 'eval_of_val_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=model_idx,
                                                                                                            model_type=model_type)
        self._new_data_eval_file = 'eval_of_new_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=model_idx,
                                                                                                            model_type=model_type)

        self._num_train_epochs = num_train_epochs
        self._train_image_dir = train_image_dir
        self._train_annotations = train_annotations
        self._val_annotations = val_annotations
        self._next_gen_annotations = next_gen_annotations

    def fit(self):
        os.system(TRAIN_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                       num_train_epochs=self._num_train_epochs,
                                       train_image_dir=self._train_image_dir,
                                       train_annotations=self._train_annotations,
                                       val_image_dir=self._val_image_dir,
                                       val_annotations=self._val_annotations,
                                       model_output_file=self._model_output_file))

    def create_val_score(self, metric='oks'):
        """
        :param metric: Metric for evaluation of the model after training
        :return: Average score for all of the training epochs
        """
        if metric == 'oks':
            checkpoint = self._model_output_file
            os.system(EVAL_OTHER_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                model_output_file=checkpoint,
                                                dataset_image_dir=self._train_image_dir,
                                                dataset_annotations=self._val_annotations,
                                                eval_output_file=self._eval_output_file))

    def select_new_images(self):
        # TODO - select images from teacher predictions (using self._model_eval_file)
        pass

    def create_new_data_scores_and_annotations(self):
        """
        :param metric: Metric for evaluation of the model after training
        :return: Average score for all of the training epochs
        """
        if os.path.exists(self._next_gen_annotations):
            os.system(EVAL_OTHER_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                model_output_file=self._model_output_file,
                                                dataset_image_dir=self._train_image_dir,
                                                dataset_annotations=self._next_gen_annotations,
                                                eval_output_file=self._new_data_eval_file))
        else:
            logging.info('next_gen_annotations file does not exist')

        # TODO create new annotations file

    def save_results(self):
        eval_output_stats_file = self._eval_output_file + '.stats.json'
        new_data_eval_stats_file = self._new_data_eval_file + '.stats_json'

        files_list = [eval_output_stats_file, new_data_eval_stats_file]
        for file in files_list:
            cmd = 'git add ' + file + ' && ' \
                + 'git commit -m "add stats file' + \
                ' && ' + 'git push origin {branch_name}'.format(branch_name='noisy-student-flow')
            os.system(cmd)

    def upload_data_to_tensorboard(self):
        # TODO
        pass
