from model import Model


class Teacher(Model):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, val_image_dir, val_annotations, next_gen_annotations, full_data_model=False):
        super().__init__(model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, val_image_dir, val_annotations, next_gen_annotations, full_data_model)
