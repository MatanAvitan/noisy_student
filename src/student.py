from src.model import Model
import os


class Student(Model):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations):
        super(Teacher, self).__init__(model_type, model_idx, num_train_epochs, train_image_dir, train_annotations)

    def inject_noise(self):
        """
        Inject noise to the input images(x)
        it's an inplace operation over self._x
        """

        # TODO: add augmentations
        pass
