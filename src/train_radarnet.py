import argparse
from radarnet_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to file that contains training paths for images')
parser.add_argument('--train_radar_path',
    type=str, required=True, help='Path to file that contains paths to radar point')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Path to file that contains ground truth lidar maps paths')
parser.add_argument('--val_image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--val_radar_path',
    type=str, required=True, help='Path to file that contains validation radar points')
parser.add_argument('--val_ground_truth_path',
    type=str, required=True, help='Path to file that contains ground truth lidar maps paths')

# Input settings
parser.add_argument('--batch_size',
    type=int, default=64, help='Batch Size for the input')
parser.add_argument('--patch_size',
    nargs='+', type=int, default=[768, 288], help='Height, width of input patch')
parser.add_argument('--total_points_sampled',
    type=int, required=True, default=4, help='Total points sampled from all the available radar points in a scene')
parser.add_argument('--sample_probability_lidar',
    type=float, required=True, default=0.0, help='Sample lidar points and add noise instead of radar points with this probability')
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input channels for the image')
parser.add_argument('--input_channels_depth',
    type=int, default=3, help='Number of input channels for the depth')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=['radarnetv1', 'batch_norm'], help='Encoder type')
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=[32, 64, 128, 128, 128], help='Number of filters per layer')
parser.add_argument('--n_neurons_encoder_depth',
    nargs='+', type=int, default=[32, 64, 128, 128, 128], help='Number of neurons per layer')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=['multiscale', 'batch_norm'], help='Decoder type')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 64, 32, 16], help='Number of filters per layer')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='kaiming_uniform', help='Range of image intensities after normalization')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Range of image intensities after normalization')

# Training Settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2e-4, 1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[25, 50, 100], help='Space delimited list to change learning rate')

# Augmentation Settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default=['none'], help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type for augmentation: none, horizontal, vertical')

# Loss settings
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight of weight decay')
parser.add_argument('--w_positive_class',
    type=float, default=1.0, help='Weight of positive class')
parser.add_argument('--max_distance_correspondence',
    type=float, default=0.4, help='Max distance to consider two points correspondence')
parser.add_argument('--set_invalid_to_negative_class',
    action='store_true', help='If set then any invalid locations are treated as negative class')

# Checkpoint and summary settings
parser.add_argument('--checkpoint_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=100, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=100, help='Number of samples to include in visual display summary')
parser.add_argument('--start_step_validation',
    type=int, default=100, help='Step to start performing validation')
parser.add_argument('--restore_path',
    type=str, default=None, help='Path to restore model from checkpoint')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')

# Hardware
parser.add_argument('--n_thread',
    type=int, default=10, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    train(train_image_path=args.train_image_path,
          train_radar_path=args.train_radar_path,
          train_ground_truth_path=args.train_ground_truth_path,
          val_image_path=args.val_image_path,
          val_radar_path=args.val_radar_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Input settings
          batch_size=args.batch_size,
          patch_size=args.patch_size,
          total_points_sampled=args.total_points_sampled,
          sample_probability_of_lidar=args.sample_probability_lidar,
          normalized_image_range=args.normalized_image_range,
          # Network settings
          encoder_type=args.encoder_type,
          n_filters_encoder_image=args.n_filters_encoder_image,
          n_neurons_encoder_depth=args.n_neurons_encoder_depth,
          decoder_type=args.decoder_type,
          n_filters_decoder=args.n_filters_decoder,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          # Loss settings
          w_weight_decay=args.w_weight_decay,
          w_positive_class=args.w_positive_class,
          max_distance_correspondence=args.max_distance_correspondence,
          set_invalid_to_negative_class=args.set_invalid_to_negative_class,
          # Checkpoint and summary settings
          checkpoint_dirpath=args.checkpoint_dirpath,
          n_step_per_summary=args.n_step_per_summary,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          start_step_validation=args.start_step_validation,
          restore_path=args.restore_path,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Hardware settings
          n_thread=args.n_thread)
