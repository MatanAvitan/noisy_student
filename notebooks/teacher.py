import os
import torch
import openpifpaf
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path('../')
DATASETS_PATH = BASE_DIR.joinpath('datasets')
DATASET_NAME = 'data-mscoco'
COCO_DATASET_PATH = DATASETS_PATH.joinpath(DATASET_NAME)
COCO_VAL_PATH = COCO_DATASET_PATH.joinpath('images/val2017')

class Teacher:
    def __init__(self,
                 teacher=None,
                 output=None,
                 data_percentage=None,
                 unlabel_data_dir=None,
                 checkpoint=None):
        self.teacher=teacher
        self.output = output if output else 'temp_output'
        if not output:
            os.mkdir(self.output)
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
            os.system("python -m openpifpaf.train \
                      --lr=1e-3 \
                      --momentum=0.95 \
                      --epochs=150 \
                      --lr-decay 120 140 \
                      --batch-size=16 \
                      --basenet=resnet101 \
                      --head-quad=1 \
                      --headnets pif paf paf25 \
                      --square-edge=401 \
                      --lambdas 10 1 1 15 1 1 15 1 1 \
                      --output {}".format(self.output))
            net_cpu, _ = openpifpaf.network.factory(checkpoint=self.output)
            self.net = net_cpu.to(device)

    def teach(self):
        # create pseudo labels

        openpifpaf.decoder.CifSeeds.threshold = 0.5
        openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
        openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
        processor = openpifpaf.decoder.factory_decode(self.net.head_nets,
                                                          basenet_stride=self.net.base_net.stride)
        from openpifpaf.datasets import CocoKeypoints
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

            annIds = coco_kps.getAnnIds(catIds=catIds)
            anns = coco_kps.loadAnns(annIds)
            predictions = processor.annotations(fields=fields_batch[0])

            # initialize COCO api for person keypoints annotations
            coco_kps=COCO(annFile)
            ec = EvalCoco(coco_kps, processor, preprocess.annotations_inverse)

            ec.from_predictions(predictions,
                                    meta_batch[0],
                                    debug=False,
                                    gt=ann_ids,
                                    image_cpu=images_batch)

