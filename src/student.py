from src.model import Model
from consts import models, DOWNLOAD_COMMAND, TRAIN_COMMAND
import os


class Student(Model):
    def __init__(self, images_dir_path, annotations_path, model_name):
        super(Model, self).__init__(images_dir_path, annotations_path, model_name)

    def fit(self, num_epochs=5):
        os.system(models[self._model_name][DOWNLOAD_COMMAND])
        os.system(models[self._model_name][TRAIN_COMMAND].format(num_epochs))
        # todo: check how can we wait here until the training stops,
        # todo: because this is not a python command but a shell command,
        # todo: it will run the command and move on.

    def inject_noise(self):
        """
        Inject noise to the input images(x)
        it's an inplace operation over self._x
        """
        pass

    def get_score(self, metric='oks'):
        """
        :param metric: Metric for evaluation of the model after training
        :return: Average score for all of the training epochs
        """
        if metric == 'oks':
            pass
