#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_nuscenes_radarnet.py \
--restore_path \
    trained_radarnet/roi_pooled_v1msbn_6x900x288_lr0-2e4_200_aug0-100_200_bri080-120_con080-120_sat080-120_hflip_wpos200_inv_as_neg_sample_lid_0_interp_gt/model-195000.pth \
--patch_size 900 288 \
--input_channels_image 3 \
--input_channels_depth 3 \
--normalized_image_range 0 1 \
--encoder_type radarnetv1 batch_norm \
--n_filters_encoder_image 32 64 128 128 128 \
--n_neurons_encoder_depth 32 64 128 128 128 \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 128 64 32 16 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--run_evaluation \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \