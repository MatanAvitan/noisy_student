import os
import json
import logging
from consts import TRAIN_COMMAND, EVAL_OTHER_COMMAND, OPENPIFPAF_PATH, ANNOTATIONS_SCORE_THRESH, OPENPIFPAF_PATH

class Model(object):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, val_image_dir, val_annotations, next_gen_annotations):
        self._model_type = model_type
        self._model_idx = model_idx
        self._model_output_file = 'model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                        model_type=self._model_type)
        self._eval_output_file = 'eval_of_val_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                                            model_type=self._model_type)
        self._new_data_eval_file = 'eval_of_new_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                                            model_type=self._model_type)

        self._num_train_epochs = num_train_epochs
        self._train_image_dir = train_image_dir
        self._train_annotations = train_annotations
        self._val_image_dir = val_image_dir
        self._val_annotations = val_annotations
        self._next_gen_annotations = next_gen_annotations

    def fit(self):
        train_process_return_value = os.system(TRAIN_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                                    num_train_epochs=self._num_train_epochs,
                                                                    train_image_dir=self._train_image_dir,
                                                                    train_annotations=self._train_annotations,
                                                                    val_image_dir=self._val_image_dir,
                                                                    val_annotations=self._val_annotations,
                                                                    model_output_file=self._model_output_file))
        logging.info('train_process_return_value:{}'.format(train_process_return_value))

    def create_val_score(self, metric='oks'):
        """
        creates val score files for val data
        """
        if metric == 'oks':
            checkpoint = self._model_output_file
            eval_process_return_value = os.system(EVAL_OTHER_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                                            model_output_file=checkpoint,
                                                                            dataset_image_dir=self._val_image_dir,
                                                                            dataset_annotations=self._val_annotations,
                                                                            eval_output_file=self._eval_output_file))
            logging.info('eval_process_return_value:{}'.format(eval_process_return_value))

    def select_new_images(self, thresh=ANNOTATIONS_SCORE_THRESH):
        logging.info('Loading new annotation file created by teacher')
        new_data_eval_pred_file_path = os.path.join(OPENPIFPAF_PATH, self._new_data_eval_file + '.pred.json')
        with open(new_data_eval_pred_file_path, 'r') as j:
            new_annotations_data = json.loads(j.read())
        logging.info('Filtering new annotation file')
        new_annotations_data_filtered_by_score = [ann for ann in new_annotations_data if ann['score'] >= thresh]
        selected_ann_data = {'annotations': []}
        total_new_annotations_filtered_count = len(new_annotations_data_filtered_by_score)
        for idx, ann in enumerate(new_annotations_data_filtered_by_score):
            logging.info('Adding annotation no.{} out of {}'.format(idx, total_new_annotations_filtered_count))
            ann['num_keypoints'] = sum([1 for i in ann['keypoints'] if i > 0]) / 3
            selected_ann_data['annotations'].append(ann)
        self._selected_ann_data = selected_ann_data

    def merge_annotations(self):
        with open(os.path.join(OPENPIFPAF_PATH, self._train_annotations), 'r') as j:
            train_ann_data = json.loads(j.read())
        for key, value in train_ann_data.items():
            logging.info('merging key: {}'.format(key))
            selected_ann_value = self._selected_ann_data.get(key,None)
            if selected_ann_value:
                if isinstance(selected_ann_value, list):
                    value.extend(selected_ann_value)
                else :
                    value.append(selected_ann_value)
        merged_file_name = 'train_annotaions_of_model_no_{model_idx}'.format(model_idx=self._model_idx+1)
        logging.info('Dumping File: {}'.format(merged_file_name))
        with open(merged_file_name, 'w') as outfile:
            json.dump(train_ann_data, outfile)
        self._merged_annotations_path = merged_file_name

    def create_new_data_scores_and_annotations(self):
        """
        Creates next gen annotations and merges them with train annotations
        """
        if os.path.exists(self._next_gen_annotations):
            eval_process_new_data_return_value = os.system(EVAL_OTHER_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                model_output_file=self._model_output_file,
                                                dataset_image_dir=self._train_image_dir,
                                                dataset_annotations=self._next_gen_annotations,
                                                eval_output_file=self._new_data_eval_file))
            logging.info('eval_process_new_data_return_value:{}'.format(eval_process_new_data_return_value))
            logging.info('select new images')
            self.select_new_images()
            logging.info('merging annotations')
            self.merge_annotations()
        else:
            logging.info('next_gen_annotations file does not exist')

    def save_results(self):
        # TODO
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
