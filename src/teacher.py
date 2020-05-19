import os
import torch
import openpifpaf
from pathlib import Path
from pycocotools.coco import COCO
from openpifpaf.datasets import CocoKeypoints
from openpifpaf import transforms as openpifpaf_transforms
from openpifpaf.eval_coco import EvalCoco

from src.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path('../../')
DATASETS_PATH = BASE_DIR.joinpath('datasets')
DATASET_NAME = 'data-mscoco'
COCO_DATASET_PATH = DATASETS_PATH.joinpath(DATASET_NAME)
COCO_VAL_PATH = COCO_DATASET_PATH.joinpath('images/val2017')
annFile = COCO_DATASET_PATH.joinpath('annotations/person_keypoints_val2017.json')


class Teacher(Model):
    def __init__(self,
                 images_dir_path, annotations_path, model_name,
                 data_percentage_to_train_on,
                 checkpoint,
                 is_first_running=True):
        super(Teacher, self).__init__(images_dir_path, annotations_path, model_name)
        self.data_percentage_to_train_on = data_percentage_to_train_on
        self.checkpoint = checkpoint
        self._is_first_running = is_first_running
        self.net = None

    def fit(self, first_running=False):
        if self.checkpoint:
            net_cpu, _ = openpifpaf.network.factory(checkpoint=self.checkpoint)
            self.net = net_cpu.to(device)

    def predict(self):
        # create pseudo labels

        decode = openpifpaf.decoder.factory_decode(self.net,
                                                   seed_threshold=0.5)
        processor = openpifpaf.decoder.Processor(self.net, decode,
                                                 instance_threshold=0.2,
                                                 keypoint_threshold=0.3)

        preprocess = openpifpaf_transforms.Compose([
            openpifpaf_transforms.NormalizeAnnotations(),
            openpifpaf_transforms.RescaleAbsolute(641),
            openpifpaf_transforms.EVAL_TRANSFORM,
        ])
        coco_person_val_ds = CocoKeypoints(root=COCO_VAL_PATH,
                                           annFile=annFile,
                                           preprocess=preprocess,
                                           all_persons=True,
                                           all_images=False)

        loader = torch.utils.data.DataLoader(coco_person_val_ds,
                                             batch_size=1,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

        for i, (images_batch, ann_ids, meta_batch) in enumerate(loader):
            images_batch = images_batch.to(device)
            fields_batch = processor.fields(images_batch)
            predictions = processor.annotations(fields=fields_batch[0])

            ############################################################################################
            # We need to save annotations and add path path to teacher, and also choose only relevant annotations
            ############################################################################################

            # initialize COCO api for person keypoints annotations
            coco_kps = COCO(annFile)
            ec = EvalCoco(coco_kps, processor, preprocess.annotations_inverse)

            from_pred_results = ec.from_predictions(predictions,
                                                    meta_batch[0],
                                                    debug=False,
                                                    gt=ann_ids,
                                                    image_cpu=images_batch)
            print(from_pred_results)
            break

    def get_score(self, metric='certainty'):
        """
        Needed to be implemented.
        :param metric:
        :return:
        """
        pass
