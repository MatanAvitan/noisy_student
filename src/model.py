from abc import ABC, abstractmethod
from consts import OUTPUT_DIR, EVAL_DIR
import os

class Model(ABC):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations):
        self._model_type = model_type
        self._model_idx = model_idx
        self._model_output_file = os.path.join(OUTPUT_DIR,
                                               'model_type_{model_type}_model_no_{model_idx}'.format(model_idx=model_idx,
                                                                                                     model_type=model_type))
        self._eval_output_file = os.path.join(EVAL_DIR,
                                              'eval_of_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=model_idx,
                                                                                                            model_type=model_type))
        self._num_train_epochs = num_train_epochs
        self._train_image_dir = train_image_dir
        self._train_annotations = train_annotations

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_score(self, metric):
        pass
