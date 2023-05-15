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
parser.add_argument('--path_to_pickle_file_train',
    type=str, required=True, help='Path to pickle file that contains training data and paths')
parser.add_argument('--path_to_pickle_file_val',
    type=str, required=True, help='Path to pickle file that contains validation data and paths')
# Input settings
parser.add_argument('--batch_size',
    type=int, default=64, help='Batch Size for the input')
parser.add_argument('--patch_size',
    nargs='+', type=int, default=[900, 60], help='Size of the image patch to be input in the model. Should be of the format (Y,X)')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')



args = parser.parse_args()

if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    # Checkpoint settings
    args.restore_path = None if args.restore_path == '' else args.restore_path

    train(path_to_pickle_file_train=args.path_to_pickle_file_train,
          path_to_pickle_file_val=args.path_to_pickle_file_val,
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
