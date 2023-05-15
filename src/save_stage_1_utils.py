import os, time, warnings
import numpy as np

# Dependencies for network, loss, etc.
import torch, torchvision
import torchvision.transforms.functional as functional
import eval_utils, losses
from models import FusionNet
from typing import NamedTuple

# Dependencies for data loading
from data_utils import Data_Utilities, save_depth
from dataset import SaveStage1OutputDataset
from transforms import Transforms

# Dependencies for logging
import log_utils
from log_utils import log
from pathlib import Path
import time
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import warnings

import random
from tqdm import tqdm
warnings.filterwarnings("ignore")
    
def run(model,
        dataloader,
        transforms,
        step=0,
        min_evaluate_depth=0.0,
        max_evaluate_depth=100.0,
        device=torch.device('cuda:0'),
        log_path=None,
        dirpath_to_save=None,
        do_evaluation=False):

    output_paths = []
    response_paths = []
    
    n_sample = len(dataloader)

    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    mae_intersection = np.zeros(n_sample)
    rmse_intersection = np.zeros(n_sample)
    imae_intersection = np.zeros(n_sample)
    irmse_intersection = np.zeros(n_sample)

    n_valid_points_output = np.zeros(n_sample)
    n_valid_points_ground_truth = np.zeros(n_sample)
    n_valid_points_intersection = np.zeros(n_sample)

    for sample_idx, batch_data in enumerate(dataloader):

        batch_data[:-1] = [
            data.to(device) for data in batch_data[:-1]
        ]

        radar_points, image, ground_truth, patch_size, pad_size, image_name = batch_data

        image = image.to(device)
        radar_points = radar_points.to(device)
        pad_size = pad_size.to(device)
        patch_size = patch_size.to(device)
        ground_truth = ground_truth.to(device)

        patch_size = patch_size.squeeze()
        pad_size = pad_size.squeeze()

        [radar_points], [image] = transforms.transform(
            points_arr=[radar_points],
            images_arr=[image],
            random_transform_probability=0.0)

        image = functional.pad(
            image,
            (pad_size, 0, pad_size, 0),
            padding_mode='edge',
            fill=0)
        image = torch.squeeze(image, 0)

        radar_points = torch.squeeze(radar_points, 0)

        output_tiles = []
        if radar_points.dim() == 2:
            radar_points = torch.unsqueeze(radar_points, 0)

        # Radar points have been repeated to N x P x 3, select the first to get N x 3
        radar_points = radar_points[:, 0, :]

        x_shifts = radar_points[:, 0].clone() + pad_size
        radar_points[:, 0] = pad_size

        # Crop the image
        image_crops = []

        height = image.shape[-2]
        crop_height = height - patch_size[0]

        for x in x_shifts:
            image_crop = image[:, crop_height:, int(x)-pad_size:int(x)+pad_size]
            image_crops.append(image_crop)

        # N x 3 x h x w image crops
        image_crops = torch.stack(image_crops, dim=0)

        output_crops = torch.sigmoid(
            model(image_crops.float(), radar_points.float(), patch_size))

        for output_crop, x in zip(output_crops, x_shifts):
            output = torch.zeros([1, image.shape[-2], image.shape[-1]] , device=device)

            # Thresholding
            output_crop = torch.where(output_crop < 0.5, torch.zeros_like(output_crop), output_crop)
            # Add crop to output
            output[:, crop_height:, int(x)-pad_size:int(x)+pad_size] = output_crop
            output_tiles.append(output)

        output_tiles = torch.cat(output_tiles, dim=0)
        output_tiles = output_tiles[:, :, pad_size:-pad_size]

        # Find the max response over all tiles
        response, output = torch.max(output_tiles, dim=0, keepdim=True)

        # Fill in the map based on z value of the points chosen
        for point_idx in range(radar_points.shape[0]):
            output = torch.where(
                output == point_idx,
                torch.full_like(output, fill_value=radar_points[point_idx, 2]),
                output)

        # Leave as 0s if we did not predict
        output = torch.where(
            torch.max(output_tiles, dim=0, keepdim=True)[0] == 0,
            torch.zeros_like(output),
            output)

        # Do evaluation against ground truth here
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        output = np.squeeze(output.cpu().numpy())
        response = np.squeeze(response.cpu().numpy())

        if dirpath_to_save is not None:
            output_radar_image_file_name = os.path.splitext(image_name[0])[0] + '.png'
            output_radar_depth_output_path = os.path.join(dirpath_to_save, output_radar_image_file_name)
            response_output_dir = os.path.join(dirpath_to_save, 'responses')
            response_output_path = os.path.join(response_output_dir, output_radar_image_file_name)

            # In case multiple threads create same directory
            if not os.path.exists(dirpath_to_save):
                try:
                    os.makedirs(dirpath_to_save)
                except:
                    pass
                
            # In case multiple threads create same directory
            if not os.path.exists(response_output_dir):
                try:
                    os.makedirs(response_output_dir)
                except:
                    pass

            save_depth(output, output_radar_depth_output_path)
            save_depth(response, response_output_path)
            output_paths.append(output_radar_depth_output_path)
            response_paths.append(response_output_path)

        if do_evaluation:
            # Validity map of output -> locations where output is valid
            validity_map_output = np.where(output > 0, 1, 0)
            validity_map_gt = np.where(ground_truth > 0, 1, 0)
            validity_map_intersection = validity_map_output*validity_map_gt

            n_valid_points_intersection[sample_idx] = np.sum(validity_map_intersection)
            n_valid_points_output[sample_idx] = np.sum(validity_map_output)
            n_valid_points_ground_truth[sample_idx] = np.sum(validity_map_gt)

            # Select valid regions to evaluate
            min_max_mask = np.logical_and(
                ground_truth > 0,
                ground_truth < 100)
            mask = np.where(np.logical_and(validity_map_gt, min_max_mask) > 0)
            mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

            output_depth_all = output[mask]
            ground_truth_all = ground_truth[mask]

            output_depth_intersection = output[mask_intersection]
            ground_truth_intersection = ground_truth[mask_intersection]

            # Compute validation metrics
            mae[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_all, 1000.0 * ground_truth_all)
            rmse[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_all, 1000.0 * ground_truth_all)
            imae[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_all, 0.001 * ground_truth_all)
            irmse[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_all, 0.001 * ground_truth_all)

            # Compute validation metrics for intersection
            mae_intersection[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
            rmse_intersection[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
            imae_intersection[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)
            irmse_intersection[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)

    if do_evaluation:
        n_valid_points_output = np.mean(n_valid_points_output)
        n_valid_points_intersection = np.mean(n_valid_points_intersection)
        n_valid_points_ground_truth = np.mean(n_valid_points_ground_truth)

        # Compute mean metrics
        mae   = np.mean(mae)
        rmse  = np.mean(rmse)
        imae  = np.mean(imae)
        irmse = np.mean(irmse)

        # Compute mean metrics for intersection
        mae_intersection = mae_intersection[~np.isnan(mae_intersection)]
        rmse_intersection = rmse_intersection[~np.isnan(rmse_intersection)]
        imae_intersection = imae_intersection[~np.isnan(imae_intersection)]
        irmse_intersection = irmse_intersection[~np.isnan(irmse_intersection)]

        mae_intersection = np.mean(mae_intersection)
        rmse_intersection = np.mean(rmse_intersection)
        imae_intersection = np.mean(imae_intersection)
        irmse_intersection = np.mean(irmse_intersection)


        # Print validation results to console
        log('Validation results (full):', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            step, mae, rmse, imae, irmse),
            log_path)

        log('Validation results (intersection):', log_path)
        log('{:>8}  {:>16}  {:>16}  {:>16}'.format(
            '', '# Ground truth', '# Output', '# Intersection'),
            log_path)
        log('{:8}  {:16.3f}  {:16.3f}  {:16.3f}'.format(
            '', n_valid_points_ground_truth, n_valid_points_output, n_valid_points_intersection),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            '', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            '', mae_intersection, rmse_intersection, imae_intersection, irmse_intersection),
            log_path)

    return output_paths, response_paths