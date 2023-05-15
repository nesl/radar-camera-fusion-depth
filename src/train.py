import argparse
from main import train
import numpy as np

from typing import NamedTuple

class Data_Struct(NamedTuple):
    scene_id: int
    sample_idx: int
    image_path: str
    ground_truth_points: np.ndarray
    input_points: np.ndarray
    ground_truth_label_path: str
    ground_truth_depth_path: str


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--path_to_pickle_file_gt_train_paths',
    type=str, required=True, help='Path to pickle file that contains training paths for lidar gt files')
parser.add_argument('--path_to_pickle_file_radar_train_numpys',
    type=str, required=True, help='Path to pickle file that contains training paths for radar training numpy files')
parser.add_argument('--path_to_pickle_file_gt_val_paths',
    type=str, required=True, help='Path to pickle file that contains validation paths for lidar gt files')
parser.add_argument('--path_to_pickle_file_radar_val_numpys',
    type=str, required=True, help='Path to pickle file that contains validation paths for radar numpy files')
parser.add_argument('--data_path',
    type=str, required=True, help='Path to where all the data is stored')
parser.add_argument('--image_path',
    type=str, required=True, help='Path to where nuScenes images are stored')
parser.add_argument('--epsilon',
    type=float, required=True, help='Depth difference between radar and lidar points')
# Input settings
parser.add_argument('--batch_size',
    type=int, default=64, help='Batch Size for the input')
parser.add_argument('--patch_size',
    nargs='+', type=int, default=[900, 60], help='Size of the image patch to be input in the model. Should be of the format (Y,X)')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')
# Training Settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[5e-5, 1e-4, 2e-4, 1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[2, 5, 10, 12, 15], help='Space delimited list to change learning rate')
# Augmentation Settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Random brightness adjustment for data augmentation')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default=['none'], help='Noise to add for augmentation: none, gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=0.0, help='Standard deviation for gaussian noise and range for uniform noise')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type for augmentation: none, horizontal, vertical')
# Loss Settings
parser.add_argument('--w_cross_entropy',
    type=float, default=1.0, help='Weight of cross entropy loss')
parser.add_argument('--w_smoothness',
    type=float, default=0.0, help='Weight of local smoothness loss')
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight of weight decay regularization for depth')
parser.add_argument('--kernel_size_smoothness',
    nargs='+', type=int, default=[11, 3], help='Weight of kernel size smoothness loss')
parser.add_argument('--set_invalid_to_negative',
    action='store_true')
parser.add_argument('--w_positive_class',
    type=float, default=1, help='Weight of positive class')
# Checkpoint and summary settings
parser.add_argument('--checkpoint_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--num_step_per_checkpoint',
    type=int, default=100, help='Number of iterations for each checkpoint')
parser.add_argument('--num_step_per_summary',
    type=int, default=100, help='Number of samples to include in visual display summary')
parser.add_argument('--start_step_validation',
    type=int, default=100, help='Step to start performing validation')
parser.add_argument('--restore_path',
    type=str, default=None, help='Path to restore model from checkpoint')
# Hardware and debugging
parser.add_argument('--debug',
    action='store_true', help='Debug mode on or off')
parser.add_argument('--num_workers',
    type=int, default=10, help='Number of threads for fetching')



args = parser.parse_args()

if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    # Checkpoint settings
    args.restore_path = None if args.restore_path == '' else args.restore_path

    train(gt_train_paths=args.path_to_pickle_file_gt_train_paths,
          radar_train_paths=args.path_to_pickle_file_radar_train_numpys,
          gt_val_paths=args.path_to_pickle_file_gt_val_paths,
          radar_val_paths=args.path_to_pickle_file_radar_val_numpys,
          data_path=args.data_path,
          image_path=args.image_path,
          epsilon=args.epsilon,
          # Input settings
          batch_size=args.batch_size,
          patch_size=args.patch_size,
          normalized_image_range=args.normalized_image_range,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          # Loss settings
          w_cross_entropy=args.w_cross_entropy,
          w_smoothness=args.w_smoothness,
          w_weight_decay=args.w_weight_decay,
          kernel_size_smoothness=args.kernel_size_smoothness,
          set_invalid_to_negative=args.set_invalid_to_negative,
          w_positive_class=args.w_positive_class,
          # Checkpoint and summary settings
          checkpoint_dirpath=args.checkpoint_dirpath,
          num_step_per_summary=args.num_step_per_summary,
          num_step_per_checkpoint=args.num_step_per_checkpoint,
          start_step_validation=args.start_step_validation,
          restore_path=args.restore_path,
          # Hardware and debugging
          debug=args.debug,
          num_workers=args.num_workers)
