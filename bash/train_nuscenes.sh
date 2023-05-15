#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
--path_to_pickle_file_train train_nuScenes_dataset_lidar_maps_interpolated_merged_5_5_100_with_filter.pkl \
--path_to_pickle_file_val val_nuScenes_dataset_lidar_maps_interpolated_merged_5_5_100_with_filter.pkl \
--batch_size 64 \
--patch_size 900 60 \
--normalized_image_range 0 1 \
--learning_rates 5e-5 1e-4 2e-4 1e-4 5e-5 \
--learning_schedule 2 5 10 12 15 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness -1 -1 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread 0.0 \
--augmentation_random_flip_type none \
--w_cross_entropy 1.00 \
--w_smoothness 1e-7 \
--w_weight_decay 0.00 \
--kernel_size_smoothness 11 3 \
--checkpoint_dirpath trained_model/model \
--num_step_per_checkpoint 1 \
--num_step_per_summary 1 \
--start_step_validation 1 \
--num_workers 23 \

