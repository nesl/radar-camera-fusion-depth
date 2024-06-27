# Depth Estimation from Camera Image and mmWave Radar Point Cloud

Pytorch implementation of Depth Estimation from Camera Image and mmWave Radar Point Cloud

CVPR 2023

Models have been tested on Ubuntu 20.04 using Python 3.8, Pytorch 1.10.2+cu113

If you use this work, please cite our paper:

```
@inproceedings{singh2023depth,
  title={Depth Estimation From Camera Image and mmWave Radar Point Cloud},
  author={Singh, Akash Deep and Ba, Yunhao and Sarker, Ankur and Zhang, Howard and Kadambi, Achuta and Soatto, Stefano and Srivastava, Mani and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9275--9285},
  year={2023}
}
``` 

## Setting up your virtual environment

We will create a virtual environment with the necessary dependencies

```
virtualenv -p /usr/bin/python3 cam_rad_fusion
source cam_rad_fusion/bin/activate

pip install -r requirements.txt
```

## Setting up dataset for RadarNet

Note: Run all bash scripts from the root directory.

We use the nuScenes dataset that can be downloaded [here](https://www.nuscenes.org/nuscenes#download).

Please create a folder called `data` and place the downloaded nuScenes dataset into it.

Generate the panoptic segmentation masks using the following:
```
python setup/gen_panoptic_seg.py
```

Then run the following bash script to generate the preprocessed dataset for training and testing:

```
bash setup_dataset_nuscenes.sh
bash setup_dataset_nuscenes_test.sh
```

This will generate the training dataset in a folder called `data/nuscenes_derived`

## Training RadarNet

To train RadarNet on the nuScenes dataset, you may run

```
bash train_radarnet_nuscenes.sh
```
To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_radarnet/<model_name>
```

## Setting up dataset for FusionNet

Once RadarNet training is complete, you can run:
```
bash setup_dataset_nuscenes_radarnet.sh
bash setup_dataset_nuscenes_radarnet_test.sh
```

## Training FusionNet

To train RadarNet on the nuScenes dataset, you may run
```
bash train_fusionnet_nuscenes.sh
```
To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_fusionnet/<model_name>
```

## Downloading our pre-trained models

Our [pretrained models](https://drive.google.com/drive/folders/1ZG-2UG67VcYhWTyHdUzTHIzdfCpsG_1Q?usp=sharing) are available here for download:

```
https://drive.google.com/drive/folders/1ZG-2UG67VcYhWTyHdUzTHIzdfCpsG_1Q?usp=sharing
```

## Evaluating our models

To evaluate the pretrained Fusionnet on the nuScenes dataset, you may run:

```
bash run_fusionnet_nuscenes_test.sh
```

You may replace the restore_path and output_path arguments to evaluate your own checkpoints

## License and disclaimer 

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to [these terms and conditions](https://github.com/nesl/radar-camera-fusion-depth/blob/main/license). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu/).
