#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_radarnet.py \
--train_image_path \
    training/nuscenes/nuscenes_train_image.txt \
--train_radar_path \
    training/nuscenes/nuscenes_train_radar.txt \
--train_ground_truth_path \
    training/nuscenes/nuscenes_train_ground_truth_interp.txt \
--val_image_path \
    validation/nuscenes/nuscenes_val_image-subset.txt \
--val_radar_path \
    validation/nuscenes/nuscenes_val_radar-subset.txt \
--val_ground_truth_path \
    validation/nuscenes/nuscenes_val_ground_truth-subset.txt \
--batch_size 6 \
--patch_size 900 288 \
--total_points_sampled 4 \
--sample_probability_lidar 0.10 \
--input_channels_image 3 \
--input_channels_depth 3 \
--normalized_image_range 0 1 \
--encoder_type radarnetv1 batch_norm \
--n_filters_encoder_image 32 64 128 128 128 \
--n_neurons_encoder_depth 32 64 128 128 128 \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 128 64 32 16 \
--learning_rates  2e-4 \
--learning_schedule 200 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--augmentation_random_flip_type horizontal \
--w_weight_decay 0.0 \
--w_positive_class 2.0 \
--max_distance_correspondence 0.4 \
--checkpoint_dirpath trained_radarnet/roi_pooled_v1msbn_6x900x288_lr0-2e4_200_aug0-100_200_bri080-120_con080-120_sat080-120_hflip_wpos200_inv_as_neg_sample_lid_10_interp_gt \
--n_step_per_checkpoint 5000 \
--n_step_per_summary 5000 \
--start_step_validation 20000 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_thread 18 \
--set_invalid_to_negative \