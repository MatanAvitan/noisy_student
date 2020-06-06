from src.model import Model
from consts import DOWNLOAD_COMMAND, TRAIN_COMMAND, EVAL_COMMAND
import os


class Teacher(Model):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations):
        super(Model, self).__init__(model_type, model_idx, num_train_epochs, train_image_dir, train_annotations)

    def fit(self):
        os.system(TRAIN_COMMAND.format(num_train_epochs=self._num_train_epochs,
                                       train_image_dir=self._train_image_dir,
                                       train_annotations=self._train_annotations,
                                       model_output_file=self._model_output_file))

        # todo: check how can we wait here until the training stops,
        # todo: because this is not a python command but a shell command,
        # todo: it will run the command and move on.

    def get_score(self, metric='oks'):
        """
        :param metric: Metric for evaluation of the model after training
        :return: Average score for all of the training epochs
        """
        if metric == 'oks':
            checkpoint = self.output_file
            os.system(EVAL_COMMAND.format(model_output_file=self._model_output_file,
                                          model_eval_file=self._model_eval_file))

    def select_new_images(self):
        # TODO - select images from teacher predictions (using self._model_eval_file)
        pass

    def create_new_annotations_file(self):
        # TODO - implement creation of new annotations file in base of teacher predictions (using self._model_eval_file) and images selected using select_new_images
        pass
