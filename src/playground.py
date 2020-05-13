import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

SAMPLE_SIZE = 100
BASE_DIR = Path('../')
DATASETS_PATH = BASE_DIR.joinpath('datasets')
DATASET_NAME = 'coco'
COCO_DATASET_PATH = DATASETS_PATH.joinpath(DATASET_NAME)
COCO_VAL_PATH = COCO_DATASET_PATH.joinpath('val2017')
RESNET101_MODEL = ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                   'v0.10.0/resnet101block5-pif-paf-paf25-edge401-191012-132602-a2bf7ecd.pkl')
IMAGE_EXT = '.jpg'

annFile = COCO_DATASET_PATH.joinpath('annotations/annotations_trainval2017/annotations/person_keypoints_val2017.json')


from torchvision.datasets import CocoDetection
from torchvision import transforms

batch_size = 1

transform = transforms.Compose([
    # If i set batch_size to more than 1 than i need to resize all the images to the same size
    transforms.ToTensor()
])

from pycocotools.coco import COCO

# initialize COCO api for person keypoints annotations
annFile = COCO_DATASET_PATH.joinpath('annotations/annotations_trainval2017/annotations/person_keypoints_val2017.json')
coco_kps=COCO(annFile)

catIds = coco_kps.getCatIds(['person'])
annIds = coco_kps.getAnnIds(catIds=catIds)
anns = coco_kps.loadAnns(annIds)

coco_person_val_ds = CocoDetection(root=COCO_VAL_PATH,
                                   annFile=annFile,
                                   transform=transform)



import openpifpaf



net_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')



if torch.cuda.is_available():
    net = net_cpu.cuda()
else:
    net = net_cpu
decode = openpifpaf.decoder.factory_decode(net,
                                           seed_threshold=0.5)
processor = openpifpaf.decoder.Processor(net, decode,
                                         instance_threshold=0.2,
                                         keypoint_threshold=0.3)


batch_size = 1

transform = transforms.Compose([
    # If i set batch_size to more than 1 than i need to resize all the images to the same size
    transforms.ToTensor()
])

coco_person_val_ds = CocoDetection(root=COCO_VAL_PATH,
                                   annFile=annFile,
                                   transform=transform)

from openpifpaf.datasets import CocoKeypoints
# Instead of using the Pytorch dataset, let's use the dataset of coco from openpifpaf
coco_person_val_ds = CocoKeypoints(root=COCO_VAL_PATH,
                                   annFile=annFile)

def gen_plot():
    import io
    """Create a pyplot plot and save to buffer."""

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf




from openpifpaf.eval_coco import EvalCoco
from openpifpaf.transforms.preprocess import Preprocess
from openpifpaf import datasets
net_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')
if torch.cuda.is_available():
    net = net_cpu.cuda()
    device = torch.device('cuda:0')
else:
    net = net_cpu
    device = torch.device('cpu:0')
decode = openpifpaf.decoder.factory_decode(net,
                                           seed_threshold=0.5)
processor = openpifpaf.decoder.Processor(net, decode,
                                         instance_threshold=0.2,
                                         keypoint_threshold=0.3)

loader = torch.utils.data.DataLoader(coco_person_val_ds, batch_size=1, pin_memory=True)

keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)


for i, (images_batch, ann_ids, meta) in enumerate(loader):
    for key, val in meta.items():
        try:
            meta[key] = val.numpy()
        except:
            pass

    images_batch = images_batch.to(device)
    fields_batch = processor.fields(images_batch)

    annIds = coco_kps.getAnnIds(catIds=catIds)
    im_anns = coco_kps.loadAnns(ann_ids[0]['id'].item())
    anns = coco_kps.loadAnns(annIds)
    # a = processor.annotations
    # key_0 = [a.item() for a in ann_ids[0]['keypoints']]
    predictions = processor.annotations(fields=fields_batch[0])
    # ann_inv = Preprocess.annotations_inverse(im_anns, meta)
    ec = EvalCoco(coco_kps, processor, Preprocess.annotations_inverse)
    ############################################################################################
    # This is the problematic row.
    ############################################################################################
    ec.from_predictions(predictions, meta)
    ec.write_predictions('bla')
    ec.stats('bla', anns[0]['id'])
    # predictions[0].score()
    im_arr = np.array(images_batch[0].cpu().permute(1, 2, 0))
    with openpifpaf.show.image_canvas(im_arr) as ax:
        keypoint_painter.annotations(ax, predictions)
        plot_buf = gen_plot()
        image = Image.open(plot_buf)
        pred_image = transforms.ToTensor()(image).unsqueeze(0)
        # keypoint_painter.keypoints(ax,predictions, target)
    plt.axis('off')
    plt.imshow(im_arr)
    ann_ids_list = [id.item() for id in ann_ids]
    target = coco_kps.loadAnns(ann_ids_list)
    coco_kps.showAnns(target)
    plot_buf = gen_plot()
    image = Image.open(plot_buf)
    label_image = transforms.ToTensor()(image).unsqueeze(0)
    break
    if i == 10:
        break
