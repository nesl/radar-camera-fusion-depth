#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python src/run_radarnet.py \
--restore_path \
    trained_radarnet/roi_pooled_v1msbn_6x900x288_lr0-2e4_200_aug0-100_200_bri080-120_con080-120_sat080-120_hflip_wpos200_inv_as_neg_sample_lid_0_interp_gt/model-195000.pth \
--image_path \
    testing/nuscenes/nuscenes_test_image.txt \
--radar_path \
    testing/nuscenes/nuscenes_test_radar.txt \
--ground_truth_path \
    testing/nuscenes/nuscenes_test_ground_truth.txt \
--patch_size 900 288 \
--input_channels_image 3 \
--input_channels_depth 3 \
--normalized_image_range 0 1 \
--encoder_type radarnetv1 batch_norm \
--n_filters_encoder_image 32 64 128 128 128 \
--n_neurons_encoder_depth 32 64 128 128 128 \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 128 64 32 16 \
--output_dirpath \
    trained_radarnet/roi_pooled_v1msbn_6x900x288_lr0-2e4_200_aug0-100_200_bri080-120_con080-120_sat080-120_hflip_wpos200_inv_as_neg_sample_lid_0_interp_gt/testing_ground_truth \
--keep_input_filenames \
--verbose \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--save_outputs \