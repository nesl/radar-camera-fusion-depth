from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, copy, argparse
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import multiprocessing as mp
import matplotlib.pyplot as plt

import data_utils

import sys
import os
from functools import reduce
import argparse
import glob
import numpy as np
from PIL import Image, ImageOps

from skimage import io
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
import pickle as pkl

import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)
nusc_explorer = NuScenesExplorer(nusc)

## COCO Label (-1) https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)

save_panoptic_masks_dir = 'data/nuscenes_derived/panoptic_segmentation_masks'

# create dir if not exists
if not os.path.exists(save_panoptic_masks_dir):
    os.makedirs(save_panoptic_masks_dir)

def get_id_from_category_id(segments_info):
    '''Returns a list of all output ids to filter. Output Ids 0-8 are moving objects in COCO'
    Input:
        segments_info (dict) - output of the panoptic segmentation model that contains meta information about all the classes
    '''
    output_ids = []
    for segment_info_dict in segments_info:
        if segment_info_dict['category_id']>=0 and segment_info_dict['category_id']<=8:
            if segment_info_dict['id'] not in output_ids:
                output_ids.append(segment_info_dict['id'])
    return torch.from_numpy(np.asarray(output_ids))

def read_image(file_name, format=None):
    '''Function to read an image
    Source: https://github.com/longyunf/rc-pda/blob/de993f9ff21357af64308e42c57197e8c7307d89/scripts/semantic_seg.py#L59'''
    
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


with torch.no_grad():
    for scene_idx in range(0,850):
        current_scene = nusc.scene[scene_idx]
        first_sample_token = current_scene['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        camera_token = first_sample['data']['CAM_FRONT']
        camera_data = nusc.get('sample_data', camera_token)
        current_scene_count = 0
        while camera_data['next'] != '':
            current_scene_count = current_scene_count + 1
            camera_image_path = camera_data['filename']
            camera_image_path = os.path.join('data/nuscenes', camera_image_path)
            raw_image = read_image(camera_image_path, 'RGB')
            
            # network
            panoptic_seg, segments_info = predictor(raw_image)["panoptic_seg"]
            torch.cuda.synchronize(device)

            # filter out all non-moving classes
            panoptic_seg_filtered = torch.isin(panoptic_seg.to("cpu"),get_id_from_category_id(segments_info))
            panoptic_seg_filtered = panoptic_seg_filtered.numpy()
            
            path_seg = os.path.join(save_panoptic_masks_dir, camera_token + '.npy')
            np.save(path_seg, panoptic_seg_filtered)            
            print('compute segmentation %d/%d, scene number: %d' % ( scene_idx, 850, current_scene_count ) )
            camera_token = camera_data['next']                
            camera_data = nusc.get('sample_data', camera_token)
        # Do everything one more time for the last scene which doesn't have a 'next'. Can replace this with a do-while
        camera_image_path = camera_data['filename']
        camera_image_path = os.path.join('data/nuscenes', camera_image_path)
        raw_image = read_image(camera_image_path, 'RGB')

        # network
        panoptic_seg, segments_info = predictor(raw_image)["panoptic_seg"]
        torch.cuda.synchronize(device)

        # filter out all non-moving classes
        panoptic_seg_filtered = torch.isin(panoptic_seg.to("cpu"),get_id_from_category_id(segments_info))
        panoptic_seg_filtered = panoptic_seg_filtered.numpy()

        path_seg = os.path.join(save_panoptic_masks_dir, camera_token + '.npy')
        np.save(path_seg, panoptic_seg_filtered)            
        print('compute segmentation %d/%d, scene number: %d' % ( scene_idx, 850, current_scene_count ) )