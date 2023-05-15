import os, time, warnings
import numpy as np

# Dependencies for network, loss, etc.
import torch, torchvision
import torchvision.transforms.functional as functional
import eval_utils, losses
from networks import RadarNet
from typing import NamedTuple

# Dependencies for data loading
from data_utils import Data_Utilities, save_depth
from dataset import SaveStage1OutputDataset
from transforms import Transforms

# Dependencies for logging
import log_utils
from log_utils import log
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import warnings

import random
from tqdm import tqdm
from save_stage_1_utils import run

from main import restore_model

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ======== Parameters to change =========================    

path_to_pickle_file_gt_paths = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/gt_images_paths_train.pkl'
path_to_pickle_file_radar_numpy_paths = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_numpy_paths_train.pkl'
path_to_nuScenes_image_dir = '/home/akash/Documents/nuScenes/samples/CAM_FRONT'

file_to_save_radar_output_paths_val = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_output_paths_val.pkl'
file_to_save_radar_output_paths_train = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_output_paths_train.pkl'
file_to_save_radar_response_paths_val = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_response_paths_val.pkl'
file_to_save_radar_response_paths_train = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_response_paths_train.pkl'


file_to_save_image_paths_val = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/stage2_image_paths_val.pkl'
file_to_save_image_paths_train = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/stage2_image_paths_train.pkl'


validation_ground_truth_paths = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/gt_images_paths_val.pkl'
validation_radar_numpy_paths = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/radar_numpy_paths_val.pkl'

#model_path_dir = '/home/akash/Documents/nuScenes/kode/Second_stage/trained_models/Predartor/lr0-1e4_2-2e4_5-5e4_10-2e4_12-1e4_15_wpos150_24x640x288_invalid_on_hflip'

#models_in_dir = os.listdir(model_path_dir)

#checkpoint_paths = []
#for model in models_in_dir:
#    if 'model' in model:
#        checkpoint_paths.append(os.path.join(model_path_dir, model))

checkpoint_path = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/model-175000.pth'
checkpoint_dirpath = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/wpos_150_invalid_on'

dir_to_save_output_train = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/generated_data_train'
dir_to_save_output_val = '/home/akash/Documents/nuScenes/Radar-camera-depth-completion_mask/Second_stage/generated_data_val'

patch_size = [640, 288]

do_eval = False

# =============== End parameters to change ===============

# Set up checkpoint directory
if not os.path.exists(checkpoint_dirpath):
    os.makedirs(checkpoint_dirpath)

log_path = os.path.join(checkpoint_dirpath, 'results.txt')

open_file = open(path_to_pickle_file_radar_numpy_paths, "rb")
radar_paths = pickle.load(open_file)
open_file.close()

open_file = open(path_to_pickle_file_gt_paths, "rb")
gt_paths = pickle.load(open_file)
open_file.close()

open_file = open(validation_radar_numpy_paths, "rb")
validation_radar_paths = pickle.load(open_file)
open_file.close()

open_file = open(validation_ground_truth_paths, "rb")
validation_gt_paths = pickle.load(open_file)
open_file.close()


train_set = SaveStage1OutputDataset(
    gt_paths = gt_paths,
    radar_numpy_paths = radar_paths,
    patch_size=patch_size,
    image_dir_path=path_to_nuScenes_image_dir
)

train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=1,
    shuffle=False,
    num_workers=1
)

val_set = SaveStage1OutputDataset(
    gt_paths = validation_gt_paths,
    radar_numpy_paths = validation_radar_paths,
    patch_size=patch_size,
    image_dir_path=path_to_nuScenes_image_dir
)

val_dataloader = DataLoader(
    dataset=val_set,
    batch_size=1,
    shuffle=False,
    num_workers=1
)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build network
height, width = patch_size
latent_height = np.ceil(height / 32.0).astype(int)
latent_width = np.ceil(width / 32.0).astype(int)

n_filters_encoder_image = [32, 64, 128, 128, 128]
n_filters_encoder_depth = [32, 64, 128, 128, 128]
latent_depth = n_filters_encoder_depth[-1]

n_filters_decoder = [256, 128, 64, 32, 16]

# Model
model = RadarNet(
        input_channels_image=3,
        input_channels_depth=3,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_output_depth=latent_height * latent_width * latent_depth,
        n_filters_decoder=n_filters_decoder,
        weight_initializer='kaiming_uniform',
        activation_func='leaky_relu',
        use_batch_norm=True)

model = torch.nn.DataParallel(model)

val_transforms = Transforms(
        normalized_image_range=[0, 1])

train_transforms = Transforms(
        normalized_image_range=[0, 1])


epoch_start_time = time.time()

train_image_second_stage_paths = []
val_image_second_stage_paths = []

#for checkpoint_path in checkpoint_paths:
# Load the model
model, _, _ = restore_model(model, checkpoint_path, None)
# Validation
model.eval()

print('Evaluating check point: {}'.format(checkpoint_path))

model = model.to(device)
# model = FusionNet().to(device)

with torch.no_grad():
    
    val_output_paths, val_response_paths = run(
                        model=model,
                        dataloader=val_dataloader,
                        transforms=val_transforms,
                        min_evaluate_depth=0.0,
                        max_evaluate_depth=100.0,
                        device=device,
                        log_path=log_path,
                        dirpath_to_save=dir_to_save_output_val,
                        do_evaluation=True)

#    run(model=model,
#        dataloader=val_dataloader,
#        transforms=val_transforms,
#        min_evaluate_depth=0.0,
#        max_evaluate_depth=100.0,
#        device=device,
#        log_path=log_path,
#        dirpath_to_save=dir_to_save_output_val,
#        do_evaluation=True)


    
    open_file = open(file_to_save_radar_output_paths_val, "wb")
    pickle.dump(val_output_paths, open_file)
    open_file.close()
    
    open_file = open(file_to_save_radar_response_paths_val, "wb")
    pickle.dump(val_response_paths, open_file)
    open_file.close()
    
    assert len(val_output_paths) == len(validation_gt_paths)
    
    val_paths = zip(
        val_output_paths, 
        validation_gt_paths,
        validation_radar_paths)
    
    for output_path, ground_truth_path, radar_path in val_paths:
        output_filename = os.path.splitext(os.path.basename(output_path))[0]
        ground_truth_filename = os.path.splitext(os.path.basename(ground_truth_path))[0]
        radar_filename = os.path.splitext(os.path.basename(radar_path))[0]
        
        image_path = os.path.join(path_to_nuScenes_image_dir, ground_truth_filename + '.jpg')
        val_image_second_stage_paths.append(image_path)
        
        assert output_filename == ground_truth_filename
        assert output_filename in radar_filename
        assert os.path.exists(image_path)

    train_output_paths, train_response_paths = run(
        model=model,
        dataloader=train_dataloader,
        transforms=train_transforms,
        min_evaluate_depth=0.0,
        max_evaluate_depth=100.0,
        device=device,
        log_path=log_path,
        dirpath_to_save=dir_to_save_output_train)
    
    open_file = open(file_to_save_radar_output_paths_train, "wb")
    pickle.dump(train_output_paths, open_file)
    open_file.close()
    
    open_file = open(file_to_save_radar_response_paths_train, "wb")
    pickle.dump(train_response_paths, open_file)
    open_file.close()

    assert len(train_output_paths) == len(gt_paths)
    
    train_paths = zip(
        train_output_paths, 
        gt_paths,
        radar_paths)
    
    for output_path, ground_truth_path, radar_path in train_paths:
        output_filename = os.path.splitext(os.path.basename(output_path))[0]
        ground_truth_filename = os.path.splitext(os.path.basename(ground_truth_path))[0]
        
        radar_filename = os.path.splitext(os.path.basename(radar_path))[0]
        
        image_path = os.path.join(path_to_nuScenes_image_dir, ground_truth_filename + '.jpg')
        train_image_second_stage_paths.append(image_path)
        
        assert output_filename == ground_truth_filename
        assert output_filename in radar_filename
        assert os.path.exists(image_path)

    open_file = open(file_to_save_image_paths_train, "wb")
    pickle.dump(train_image_second_stage_paths, open_file)
    open_file.close()

    open_file = open(file_to_save_image_paths_val, "wb")
    pickle.dump(val_image_second_stage_paths, open_file)
    open_file.close()