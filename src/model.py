from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, images_dir_path, annotations_path, model_name):
        self._x = images_dir_path
        self._y = annotations_path
        self._model_name = model_name

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_score(self, metric):
        pass
