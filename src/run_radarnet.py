import argparse
from radarnet_main import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model from checkpoint')
parser.add_argument('--image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--radar_path',
    type=str, required=True, help='Path to file that contains validation radar points')
parser.add_argument('--ground_truth_path',
    type=str, required=None, help='Path to file that contains ground truth lidar maps paths')

# Input settings
parser.add_argument('--patch_size',
    nargs='+', type=int, default=[768, 288], help='Height, width of input patch')
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

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then save outputs to output directory')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original file names')
parser.add_argument('--verbose',
    action='store_true', help='If set then print progress')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')


args = parser.parse_args()

if __name__ == '__main__':

    run(restore_path=args.restore_path,
        image_path=args.image_path,
        radar_path=args.radar_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        patch_size=args.patch_size,
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
        # Output settings
        output_dirpath=args.output_dirpath,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        verbose=args.verbose,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth)
