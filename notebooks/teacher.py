import os
import torch
import openpifpaf
from pathlib import Path
from pycocotools.coco import COCO
from openpifpaf.datasets import CocoKeypoints
from openpifpaf import transforms as openpifpaf_transforms
from openpifpaf.eval_coco import EvalCoco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path('../../')
DATASETS_PATH = BASE_DIR.joinpath('datasets')
DATASET_NAME = 'data-mscoco'
COCO_DATASET_PATH = DATASETS_PATH.joinpath(DATASET_NAME)
COCO_VAL_PATH = COCO_DATASET_PATH.joinpath('images/val2017')
annFile = COCO_DATASET_PATH.joinpath('annotations/person_keypoints_val2017.json')


class Teacher:
    def __init__(self,
                 teacher=None,
                 output=None,
                 data_percentage=None,
                 unlabel_data_dir=None,
                 checkpoint=None):
        self.teacher=teacher
        self.output = output if output else 'temp_output'
        self.data_percentage=data_percentage
        self.unlabel_data_dir=unlabel_data_dir
        self.checkpoint=checkpoint
        self.net = None

    def learn(self):
        # load trained model or train model
        if self.checkpoint:
            net_cpu, _ = openpifpaf.network.factory(checkpoint=self.checkpoint)
            self.net = net_cpu.to(device)
        else:
            # train
        ############################################################################################
        # We need to train the model and eval using our OKS - in order to know when to stop
        # perhaps we can use some kind of loop - similar to what is suggested in openpifpaf:
        ############################################################################################

############################################################################################
#         while true; do \
#   CUDA_VISIBLE_DEVICES=0 find outputs/ -name "resnet101block5-pif-paf-l1-190109-113346.pkl.epoch???" -exec \
#     python3 -m openpifpaf.eval_coco --checkpoint {} -n 500 --long-edge=641 --skip-existing \; \
#   ; \
#   sleep 300; \
# done
############################################################################################

            # os.system("python -m openpifpaf.train \
            #           --lr=1e-3 \
            #           --momentum=0.95 \
            #           --epochs=150 \
            #           --lr-decay 120 140 \
            #           --batch-size=16 \
            #           --basenet=resnet101 \
            #           --head-quad=1 \
            #           --headnets pif paf paf25 \
            #           --square-edge=401 \
            #           --lambdas 10 1 1 15 1 1 15 1 1 \
            #           --output {}".format(self.output))
            net_cpu, _ = openpifpaf.network.factory(checkpoint=self.checkpoint)
            self.net = net_cpu.to(device)

    def teach(self):
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

