#!/bin/bash

python setup/setup_dataset_nuscenes_test.py \
--nuscenes_data_root_dirpath data/nuscenes \
--nuscenes_data_derived_dirpath data/nuscenes_derived_test \
--n_scenes_to_process 150 \
--n_forward_frames_to_reproject 24 \
--n_backward_frames_to_reproject 24 \
--n_thread 40
