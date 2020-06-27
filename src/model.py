import os
import json
import logging
import random
import boto3
from matplotlib.image import imread
from data_consts import OPENPIFPAF_PATH, MERGED_TRAIN_ANNOTATIONS_FILE_PREFIX
from consts import (TRAIN_COMMAND,
                    EVAL_COMMAND,
                    PREDICT_COMMAND,
                    ANNOTATIONS_SCORE_THRESH,
                    MOCK_RUN,
                    S3_BUCKET_NAME,
                    AWS_ACCESS_ID,
                    AWS_ACCESS_KEY)


class Model(object):
    def __init__(self, model_type, model_idx, num_train_epochs, train_image_dir, train_annotations, val_image_dir, val_annotations, next_gen_annotations, full_data_model=False):
        self._model_type = model_type
        self._model_idx = model_idx
        if full_data_model:
            model_output_file_suffix = '_full_training_data'
        else:
            model_output_file_suffix = ''
        self._model_output_file = 'model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                        model_type=self._model_type) + model_output_file_suffix
        self._eval_output_file = 'eval_of_val_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                                            model_type=self._model_type) + model_output_file_suffix
        self._new_data_eval_file = 'eval_of_new_dataset_model_type_{model_type}_model_no_{model_idx}'.format(model_idx=self._model_idx,
                                                                                                            model_type=self._model_type) + model_output_file_suffix

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
        logging.info('train_process_return_value:{return_value}'.format(return_value=train_process_return_value))

    def create_val_score(self, metric='oks'):
        """
        creates val score files for val data
        """
        if metric == 'oks':
            checkpoint = self._model_output_file
            eval_process_return_value = os.system(EVAL_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                                            model_output_file=checkpoint,
                                                                            dataset_image_dir=self._val_image_dir,
                                                                            dataset_annotations=self._val_annotations,
                                                                            eval_output_file=self._eval_output_file))
            logging.info('eval_process_return_value:{return_value}'.format(return_value=eval_process_return_value))

    def select_new_images(self, thresh=ANNOTATIONS_SCORE_THRESH):
        logging.info('Loading new annotation file created by teacher')
        new_data_eval_pred_file_path = os.path.join(OPENPIFPAF_PATH, self._new_data_eval_file + '.pred.json')
        with open(new_data_eval_pred_file_path, 'r') as j:
            new_annotations_data = json.loads(j.read())
        logging.info('Filtering new annotation dict')
        new_annotations_data_filtered_by_score = [ann for ann in new_annotations_data if ann['score'] >= thresh]
        logging.info('Loading train annotations')
        with open(os.path.join(OPENPIFPAF_PATH, self._train_annotations), 'r') as j:
            train_ann_data = json.loads(j.read())
        logging.info('Find max_id in train annotations')
        max_id = 0
        mock_num_keypoints = 0
        mock_keypoints = None
        for idx, ann in enumerate(train_ann_data['annotations']):
            max_id = max(max_id, ann['id'])
            if MOCK_RUN == 'TRUE' and mock_num_keypoints == 0:
                mock_keypoints = ann['keypoints']
                mock_num_keypoints = ann['num_keypoints']
        logging.info('Load original next gen annotations file for additional info')
        next_gen_annotations_path = os.path.join(OPENPIFPAF_PATH, self._next_gen_annotations)
        with open(next_gen_annotations_path, 'r') as j:
            next_gen_annotations_data = json.loads(j.read())
        file_names = {}
        flickr_urls = {}
        for image in next_gen_annotations_data['images']:
            file_names[image['id']] = image.get('file_name')
            flickr_urls[image['id']] = image.get('flickr_url')
        logging.info('Create new annotations dict from new annotations')
        selected_ann_data = {'annotations': [], 'images': []}
        total_new_annotations_filtered_count = len(new_annotations_data_filtered_by_score)
        added_images_ids = []
        for idx, ann in enumerate(new_annotations_data_filtered_by_score):
            logging.info('Adding annotation no.{idx} out of {total}'.format(idx=idx+1,
                                                                            total=total_new_annotations_filtered_count))
            if MOCK_RUN == 'TRUE' and idx % 20 == 0:
                ann['num_keypoints'] = mock_num_keypoints
                ann['keypoints'] = mock_keypoints
            else:
                ann['num_keypoints'] = sum([1 for i in ann['keypoints'] if i > 0]) / 3
            ann['id'] = max_id + 1
            max_id += 1
            # add key, value iscrowd, 0
            ann['iscrowd'] = 0
            selected_ann_data['annotations'].append(ann)
            # add image_id if not exists
            if ann['image_id'] not in added_images_ids:
                selected_ann_data['images'].append({'id': ann['image_id'],
                                                    'file_name': file_names[ann['image_id']],
                                                    'flickr_url': flickr_urls[ann['image_id']]})
                added_images_ids.append(ann['image_id'])
        self._selected_ann_data = selected_ann_data

    def merge_annotations(self):
        with open(os.path.join(OPENPIFPAF_PATH, self._train_annotations), 'r') as j:
            train_ann_data = json.loads(j.read())
        for key, value in train_ann_data.items():
            logging.info('merging key: {key}'.format(key=key))
            selected_ann_value = self._selected_ann_data.get(key)
            if selected_ann_value:
                if isinstance(selected_ann_value, list):
                    value.extend(selected_ann_value)
                else :
                    value.append(selected_ann_value)
        merged_file_name = os.path.join(OPENPIFPAF_PATH, '{prefix}_{model_idx}'.format(prefix=MERGED_TRAIN_ANNOTATIONS_FILE_PREFIX,
                                                                                       model_idx=self._model_idx+1))
        logging.info('Dumping File: {merged_file_name}'.format(merged_file_name=merged_file_name))
        with open(merged_file_name, 'w') as outfile:
            json.dump(train_ann_data, outfile)
        self._merged_annotations_path = merged_file_name

    def create_new_data_scores_and_annotations(self):
        """
        Creates next gen annotations and merges them with train annotations
        """
        if self._next_gen_annotations is not None and os.path.exists(self._next_gen_annotations):
            eval_process_new_data_return_value = os.system(EVAL_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                model_output_file=self._model_output_file,
                                                dataset_image_dir=self._train_image_dir,
                                                dataset_annotations=self._next_gen_annotations,
                                                eval_output_file=self._new_data_eval_file))
            logging.info('eval_process_new_data_return_value:{return_value}'.format(return_value=eval_process_new_data_return_value))
            logging.info('select new images')
            self.select_new_images()
            logging.info('merging annotations')
            self.merge_annotations()
        else:
            logging.info('next_gen_annotations file does not exist')

    def save_results(self, experiment_name):
        logging.info('Starting Saving Results of Model {model_idx} in S3'.format(model_idx=self._model_idx))
        eval_output_stats_file_name = self._eval_output_file + '.stats.json'
        new_data_eval_stats_file_name = self._new_data_eval_file + '.stats.json'

        eval_output_stats_file_path = os.path.join(OPENPIFPAF_PATH, eval_output_stats_file_name)
        new_data_eval_stats_file_path = os.path.join(OPENPIFPAF_PATH, new_data_eval_stats_file_name)

        files = [(eval_output_stats_file_name, eval_output_stats_file_path), (new_data_eval_stats_file_name, new_data_eval_stats_file_path)]
        s3 = boto3.resource('s3',
                            aws_access_key_id=AWS_ACCESS_ID,
                            aws_secret_access_key=AWS_ACCESS_KEY)
        for filename, filepath in files:
            if os.path.exists(filepath):
                logging.info('Uploading to Bucket {bucket_name}, Experiment {experiment_name}, Filename {filename}'.format(bucket_name=S3_BUCKET_NAME,
                                                                                                                           experiment_name=experiment_name,
                                                                                                                           filename=filename))
                s3.meta.client.upload_file(filepath, S3_BUCKET_NAME, os.path.join(experiment_name,filename))
        logging.info('Finished Saving Results of Model {model_idx} in S3'.format(model_idx=self._model_idx))

    def save_logs(self, experiment_name):
        logging.info('Starting Saving Logs of Model {model_idx} in S3'.format(model_idx=self._model_idx))
        filename = self._model_output_file + '.log'
        filepath = os.path.join(OPENPIFPAF_PATH, filename)
        s3 = boto3.resource('s3',
                            aws_access_key_id=AWS_ACCESS_ID,
                            aws_secret_access_key=AWS_ACCESS_KEY)
        if os.path.exists(filepath):
            logging.info('Uploading to Bucket {}, Experiment {}, Filename {}'.format(S3_BUCKET_NAME, experiment_name, filename))
            s3.meta.client.upload_file(filepath, S3_BUCKET_NAME, os.path.join(experiment_name,filename))
        logging.info('Finished Saving Logs of Model {model_idx} in S3'.format(model_idx=self._model_idx))

    def save_model(self, experiment_name):
        logging.info('Starting Saving Model {model_idx} in S3'.format(model_idx=self._model_idx))
        filename = self._model_output_file
        filepath = os.path.join(OPENPIFPAF_PATH, filename)
        s3 = boto3.resource('s3',
                            aws_access_key_id=AWS_ACCESS_ID,
                            aws_secret_access_key=AWS_ACCESS_KEY)
        if os.path.exists(filepath):
            logging.info('Uploading to Bucket {}, Experiment {}, Filename {}'.format(S3_BUCKET_NAME, experiment_name, filename))
            s3.meta.client.upload_file(filepath, S3_BUCKET_NAME, os.path.join(experiment_name,filename))
        logging.info('Finished Saving Model {model_idx} in S3'.format(model_idx=self._model_idx))

    def create_images_for_tb(self, experiment_name, tb_writer, tb_image_output_dir):
        logging.info('Starting image creation for TB of {model_idx} in S3'.format(model_idx=self._model_idx))
        with open(os.path.join(OPENPIFPAF_PATH, self._val_annotations), 'r') as j:
            val_ann_data = json.loads(j.read())
        # filter annotation with category_id == 1 only
        val_images_of_humans = set([ann['image_id'] for ann in val_ann_data['annotations'] if ann['category_id'] == 1])
        # get images file_names
        val_image_names_of_humans = [image['file_name'] for image in val_ann_data['images'] if image['id'] in val_images_of_humans]

        for epoch in range(1, self._num_train_epochs+1, 20):
            logging.info('Creating images predictions for TB - epoch {epoch}'.format(epoch=epoch))
            curr_model = '{}.epoch{:03d}'.format(self._model_output_file, epoch)
            curr_model_path = os.path.join(OPENPIFPAF_PATH, curr_model)
            logging.info('epoch model name: {}'.format(curr_model))
            assert os.path.exists(curr_model_path)
            random_images_names = random.sample(val_image_names_of_humans, 20)

            images_paths = [os.path.join(self._val_image_dir, image_name) \
                            for image_name in random_images_names]
            for image_path in images_paths:
                os.system(PREDICT_COMMAND.format(openpifpaf_path=OPENPIFPAF_PATH,
                                                 images=image_path,
                                                 checkpoint=curr_model,
                                                 image_output_dir=tb_image_output_dir))
            for image_name in random_images_names:
                curr_pred_image_path = os.path.join(OPENPIFPAF_PATH, tb_image_output_dir, image_name + '.predictions.png')
                img = imread(curr_pred_image_path)
                img = torch.from_numpy(np.array(img.cpu().permute(1, 2, 0)))
                image_tb_file_name = 'Experiment {}'.format(experiment_name) + \
                                      ' ' + curr_model + \
                                      ' epoch {epoch}, image {image_name}'.format(epoch=epoch,
                                                                                   image_name=curr_image_name)
                tb_writer.add_image(image_tb_file_name, img)
            logging.info('Finished images predictions TB - epoch {epoch}'.format(epoch=epoch))
        logging.info('Finished image creation for TB of {model_idx} in S3'.format(model_idx=self._model_idx))
