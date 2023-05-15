#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_fusionnet.py \
--restore_path \
    trained_fusionnet/fus18add6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_reproj/model-165000.pth \
--image_path \
    testing/nuscenes/nuscenes_test_image.txt \
--depth_path \
    testing/nuscenes/nuscenes_test_depth_predicted.txt \
--response_path \
    testing/nuscenes/nuscenes_test_response_predicted.txt \
--ground_truth_path \
    testing/nuscenes/nuscenes_test_lidar.txt \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--encoder_type fusionnet18 batch_norm \
--n_filters_encoder_image 32 64 128 256 256 \
--n_filters_encoder_depth 16 32 64 128 128 \
--fusion_type add \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 128 64 64 32 \
--n_resolutions_decoder 1 \
--min_predict_depth 1.0 \
--max_predict_depth 100.0 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--output_dirpath \
    trained_fusionnet/fus18add6ms1bn_16x448x448_lr0-1e3_100_aug0-100_100_bri080-120_con080-120_sat080-120_hflip_l1_sm000_wd000_outrm7-150_dilate0_min1max100_lidar_loss200_reproj/output/ \
--keep_input_filenames \
--verbose \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 80.0 \