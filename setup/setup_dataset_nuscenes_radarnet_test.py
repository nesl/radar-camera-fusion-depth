import os, sys, argparse
import torch
import numpy as np

sys.path.insert(0, 'src')
import data_utils, datasets, eval_utils
from log_utils import log
from radarnet_main import forward, log_network_settings, log_evaluation_results
from radarnet_model import RadarNetModel
from radarnet_transforms import Transforms


'''
Input filepaths
'''
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')

NUSCENES_DATA_ROOT_DIRPATH = os.path.join('data', 'nuscenes')

TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_image.txt')
TEST_RADAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_radar.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_ground_truth.txt')

'''
Output filepaths
'''
NUSCENES_DATA_DERIVED_ROOT_DIRPATH = os.path.join('data', 'nuscenes_derived_test')

TEST_DEPTH_PREDICTED_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_depth_predicted.txt')
TEST_RESPONSE_PREDICTED_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_response_predicted.txt')


'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

# Checkpoint path
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model from checkpoint')

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

# Evaluation settings
parser.add_argument('--run_evaluation',
    action='store_true', help='If set then run evaluation')
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')

parser.add_argument('--paths_only',
    action='store_true', help='If set then generate paths without storing outputs')


args = parser.parse_args()


'''
Main function
'''
if __name__ == '__main__':

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Read input paths
    '''
    # Load training paths
    test_image_paths = data_utils.read_paths(TEST_IMAGE_FILEPATH)
    test_radar_paths = data_utils.read_paths(TEST_RADAR_FILEPATH)
    test_ground_truth_paths = data_utils.read_paths(TEST_GROUND_TRUTH_FILEPATH)

    n_test_sample = len(test_image_paths)

    assert n_test_sample == len(test_radar_paths)
    assert n_test_sample == len(test_ground_truth_paths)

    '''
    Set up inputs and outputs
    '''
    test_depth_predicted_paths = []
    test_response_predicted_paths = []

    inputs_outputs = [
        [
            'training',
            test_image_paths,
            test_radar_paths,
            test_ground_truth_paths,
            test_depth_predicted_paths,
            test_response_predicted_paths,
            TEST_DEPTH_PREDICTED_FILEPATH,
            TEST_RESPONSE_PREDICTED_FILEPATH
        ]
    ]

    '''
    Set up the model
    '''
    # Build network
    radarnet_model = RadarNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=args.patch_size,
        encoder_type=args.encoder_type,
        n_filters_encoder_image=args.n_filters_encoder_image,
        n_neurons_encoder_depth=args.n_neurons_encoder_depth,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        device=device)

    radarnet_model.eval()
    radarnet_model.to(device)
    radarnet_model.data_parallel()

    parameters_radarnet_model = radarnet_model.parameters()

    step, _ = radarnet_model.restore_model(args.restore_path)

    log('Restoring checkpoint from: \n{}\n'.format(args.restore_path))

    log_network_settings(
        log_path=None,
        # Network settings
        encoder_type=args.encoder_type,
        n_filters_encoder_image=args.n_filters_encoder_image,
        n_neurons_encoder_depth=args.n_neurons_encoder_depth,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        parameters_model=parameters_radarnet_model)

    '''
    Process each set of input and outputs
    '''
    for paths in inputs_outputs:

        # Unpack inputs and outputs
        tag, \
            image_paths, \
            radar_paths, \
            ground_truth_paths, \
            depth_predicted_paths, \
            response_predicted_paths, \
            depth_predicted_filepath, \
            response_predicted_filepath = paths

        # Create output paths for depth and response
        for radar_path in radar_paths:

            # Create depth path and store
            depth_predicted_path = \
                radar_path.replace('radar_points', 'depth_predicted')

            depth_predicted_path = \
                os.path.splitext(depth_predicted_path)[0] + '.png'

            depth_predicted_paths.append(depth_predicted_path)

            # Create response path and store
            response_predicted_path = \
                radar_path.replace('radar_points', 'response_predicted')

            response_predicted_path = \
                os.path.splitext(response_predicted_path)[0] + '.png'

            response_predicted_paths.append(response_predicted_path)

        # Create directories
        depth_predicted_dirpaths = [
            os.path.dirname(path) for path in depth_predicted_paths
        ]

        depth_predicted_dirpaths = np.unique(depth_predicted_dirpaths)

        response_predicted_dirpaths = [
            os.path.dirname(path) for path in response_predicted_paths
        ]

        response_predicted_dirpaths = np.unique(response_predicted_dirpaths)

        for dirpaths in [depth_predicted_dirpaths, response_predicted_dirpaths]:
            for dirpath in dirpaths:
                os.makedirs(dirpath, exist_ok=True)

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(
            datasets.RadarNetInferenceDataset(
                image_paths=image_paths,
                radar_paths=radar_paths,
                ground_truth_paths=ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        transforms = Transforms(
            normalized_image_range=args.normalized_image_range)

        n_sample = len(dataloader)

        if args.run_evaluation and not args.paths_only:
            # Define evaluation metrics
            mae_intersection = np.zeros(n_sample)
            rmse_intersection = np.zeros(n_sample)
            imae_intersection = np.zeros(n_sample)
            irmse_intersection = np.zeros(n_sample)

            n_valid_points_output = np.zeros(n_sample)
            n_valid_points_ground_truth = np.zeros(n_sample)
            n_valid_points_intersection = np.zeros(n_sample)

        if not args.paths_only:
            # Iterate through data loader
            for sample_idx, data in enumerate(dataloader):

                with torch.no_grad():
                    data = [
                        datum.to(device) for datum in data
                    ]

                    image, radar_points, ground_truth = data

                    [image], [radar_points] = transforms.transform(
                        images_arr=[image],
                        points_arr=[radar_points],
                        random_transform_probability=0.0)

                    output_depth, output_response = forward(
                        model=radarnet_model,
                        image=image,
                        radar_points=radar_points,
                        device=device)

                '''
                Save outputs
                '''
                output_depth = np.squeeze(output_depth.cpu().numpy())
                output_response = np.squeeze(output_response.cpu().numpy())

                data_utils.save_depth(output_depth, depth_predicted_paths[sample_idx])
                data_utils.save_response(output_response, response_predicted_paths[sample_idx])

                print('Processed {}/{} {} samples'.format(sample_idx + 1, n_sample, tag), end='\r')

                '''
                Evaluate results
                '''
                if args.run_evaluation:
                    ground_truth = np.squeeze(ground_truth.cpu().numpy())

                    # Validity map of output -> locations where output is valid
                    validity_map_output = np.where(output_depth > 0, 1, 0)
                    validity_map_ground_truth = np.where(ground_truth > 0, 1, 0)
                    validity_map_intersection = validity_map_output * validity_map_ground_truth

                    n_valid_points_intersection[sample_idx] = np.sum(validity_map_intersection)
                    n_valid_points_output[sample_idx] = np.sum(validity_map_output)
                    n_valid_points_ground_truth[sample_idx] = np.sum(validity_map_ground_truth)

                    # Select valid regions to evaluate
                    min_max_mask = np.logical_and(
                        ground_truth > args.min_evaluate_depth,
                        ground_truth < args.max_evaluate_depth)
                    mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

                    output_depth_intersection = output_depth[mask_intersection]
                    ground_truth_intersection = ground_truth[mask_intersection]

                    # Compute validation metrics for intersection
                    mae_intersection[sample_idx] = eval_utils.mean_abs_err(
                        1000.0 * output_depth_intersection,
                        1000.0 * ground_truth_intersection)
                    rmse_intersection[sample_idx] = eval_utils.root_mean_sq_err(
                        1000.0 * output_depth_intersection,
                        1000.0 * ground_truth_intersection)
                    imae_intersection[sample_idx] = eval_utils.inv_mean_abs_err(
                        0.001 * output_depth_intersection,
                        0.001 * ground_truth_intersection)
                    irmse_intersection[sample_idx] = eval_utils.inv_root_mean_sq_err(
                        0.001 * output_depth_intersection,
                        0.001 * ground_truth_intersection)

            if args.run_evaluation:
                n_valid_points_output = np.mean(n_valid_points_output)
                n_valid_points_intersection = np.mean(n_valid_points_intersection)
                n_valid_points_ground_truth = np.mean(n_valid_points_ground_truth)

                # Compute mean metrics for intersection
                mae_intersection = mae_intersection[~np.isnan(mae_intersection)]
                rmse_intersection = rmse_intersection[~np.isnan(rmse_intersection)]
                imae_intersection = imae_intersection[~np.isnan(imae_intersection)]
                irmse_intersection = irmse_intersection[~np.isnan(irmse_intersection)]

                mae_intersection = np.mean(mae_intersection)
                rmse_intersection = np.mean(rmse_intersection)
                imae_intersection = np.mean(imae_intersection)
                irmse_intersection = np.mean(irmse_intersection)

                # Print evaluation results to console
                log_evaluation_results(
                    title='Evaluation results on {} samples from {} set'.format(n_sample, tag),
                    mae_intersection=mae_intersection,
                    rmse_intersection=rmse_intersection,
                    imae_intersection=imae_intersection,
                    irmse_intersection=irmse_intersection,
                    n_valid_points_output=n_valid_points_output,
                    n_valid_points_intersection=n_valid_points_intersection,
                    n_valid_points_ground_truth=n_valid_points_ground_truth,
                    step=step,
                    log_path=None)

        print('Storing {} {} predicted depth maps file paths into: {}'.format(
            len(depth_predicted_paths), tag, depth_predicted_filepath))
        data_utils.write_paths(depth_predicted_filepath, depth_predicted_paths)

        print('Storing {} {} predicted response maps file paths into: {}'.format(
            len(response_predicted_paths), tag, response_predicted_filepath))
        data_utils.write_paths(response_predicted_filepath, response_predicted_paths)