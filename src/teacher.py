from model import Model
import os


class Teacher(Model):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations):
        super(Teacher, self).__init__(model_type, model_idx, num_train_epochs, train_image_dir, train_annotations)


    def create_new_annotations_file(self):
        # TODO - implement creation of new annotations file in base of teacher predictions (using self._model_eval_file) and images selected using select_new_images
        pass
