#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_fusionnet.py \
--train_image_path \
    training/nuscenes/nuscenes_train_image.txt \
--train_depth_path \
    training/nuscenes/nuscenes_train_depth_predicted.txt \
--train_response_path \
    training/nuscenes/nuscenes_train_response_predicted.txt \
--train_ground_truth_path \
    training/nuscenes/nuscenes_train_ground_truth_interp.txt \
--train_lidar_map_path \
    training/nuscenes/nuscenes_train_ground_truth.txt \
--val_image_path \
    validation/nuscenes/nuscenes_val_image.txt \
--val_depth_path \
    validation/nuscenes/nuscenes_val_depth_predicted.txt \
--val_response_path \
    validation/nuscenes/nuscenes_val_response_predicted.txt \
--val_ground_truth_path \
    validation/nuscenes/nuscenes_val_lidar.txt \
--batch_size 16 \
--n_height 448 \
--n_width 448 \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--encoder_type fusionnet18 batch_norm \
--n_filters_encoder_image 32 64 128 256 256 256 \
--n_filters_encoder_depth 16 32 64 128 128 128 \
--fusion_type weight_and_project \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 256 128 64 64 32 \
--n_resolutions_decoder 1 \
--min_predict_depth 1.0 \
--max_predict_depth 100.0 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--learning_rates 1e-3 \
--learning_schedule 450 \
--loss_func l1 \
--w_smoothness 0.0 \
--w_lidar_loss 2.0 \
--w_weight_decay 0.0 \
--loss_smoothness_kernel_size -1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--ground_truth_dilation_kernel_size -1 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--augmentation_random_flip_type horizontal \
--min_evaluate_depth 0 \
--max_evaluate_depth 100 \
--checkpoint_dirpath \
    trained_fusionnet/fus18project6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_interp_with_reproj \
--n_step_per_checkpoint 5000 \
--n_step_per_summary 5000 \
--start_step_validation 25000 \
--n_thread 15 \