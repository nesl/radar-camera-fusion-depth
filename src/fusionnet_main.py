import os, time
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import data_utils, datasets, eval_utils
from log_utils import log
from fusionnet_model import FusionNetModel
from fusionnet_transforms import Transforms
from net_utils import OutlierRemoval


def train(train_image_path,
          train_depth_path,
          train_response_path,
          train_ground_truth_path,
          train_lidar_map_path,
          val_image_path,
          val_depth_path,
          val_response_path,
          val_ground_truth_path,
          # Batch settings
          batch_size,
          n_height,
          n_width,
          # Input settings
          input_channels_image,
          input_channels_depth,
          normalized_image_range,
          # Network settings
          encoder_type,
          n_filters_encoder_image,
          n_filters_encoder_depth,
          fusion_type,
          decoder_type,
          n_filters_decoder,
          n_resolutions_decoder,
          min_predict_depth,
          max_predict_depth,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_crop_type,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          augmentation_random_flip_type,
          # Loss settings
          loss_func,
          w_smoothness,
          w_weight_decay,
          loss_smoothness_kernel_size,
          w_lidar_loss,
          ground_truth_outlier_removal_kernel_size,
          ground_truth_outlier_removal_threshold,
          ground_truth_dilation_kernel_size,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_dirpath,
          n_step_per_summary,
          n_step_per_checkpoint,
          start_step_validation,
          restore_path,
          # Hardware settings
          device,
          n_thread):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    # Set up checkpoint and event paths
    depth_model_checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    train_image_paths = data_utils.read_paths(train_image_path)
    train_depth_paths = data_utils.read_paths(train_depth_path)
    train_response_paths = data_utils.read_paths(train_response_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    train_lidar_map_paths = data_utils.read_paths(train_lidar_map_path)

    n_train_sample = len(train_image_paths)

    for paths in [train_depth_paths, train_response_paths, train_ground_truth_paths, train_lidar_map_paths]:
        assert n_train_sample == len(paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetTrainingDataset(
            image_paths=train_image_paths,
            depth_paths=train_depth_paths,
            response_paths=train_response_paths,
            ground_truth_paths=train_ground_truth_paths,
            lidar_map_paths=train_lidar_map_paths,
            shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_thread)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_flip_type=augmentation_random_flip_type)

    '''
    Set up paths for validation
    '''
    val_image_paths = data_utils.read_paths(val_image_path)
    val_depth_paths = data_utils.read_paths(val_depth_path)
    val_response_paths = data_utils.read_paths(val_response_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    n_val_sample = len(val_image_paths)

    for paths in [val_depth_paths, val_response_paths, val_ground_truth_paths]:
        assert n_val_sample == len(paths)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetInferenceDataset(
            image_paths=val_image_paths,
            depth_paths=val_depth_paths,
            response_paths=val_response_paths,
            ground_truth_paths=val_ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Initialize ground truth outlier removal
    if ground_truth_outlier_removal_kernel_size > 1 and ground_truth_outlier_removal_threshold > 0:
        ground_truth_outlier_removal = OutlierRemoval(
            kernel_size=ground_truth_outlier_removal_kernel_size,
            threshold=ground_truth_outlier_removal_threshold)
    else:
        ground_truth_outlier_removal = None

    # Initialize ground truth dilation
    if ground_truth_dilation_kernel_size > 1:
        ground_truth_dilation = torch.nn.MaxPool2d(
            kernel_size=ground_truth_dilation_kernel_size,
            stride=1,
            padding=ground_truth_dilation_kernel_size // 2)
    else:
        ground_truth_dilation = None

    '''
    Set up the model
    '''
    # Build network
    fusionnet_model = FusionNetModel(
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        decoder_type=decoder_type,
        n_resolution_decoder=n_resolutions_decoder,
        n_filters_decoder=n_filters_decoder,
        deconv_type='up',
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    fusionnet_model.to(device)
    fusionnet_model.data_parallel()

    parameters_fusionnet_model = fusionnet_model.parameters()

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_path,
        train_depth_path,
        train_response_path,
        train_ground_truth_path
    ]
    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_depth_path,
        val_response_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=3,
        input_channels_depth=3,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolutions_decoder=n_resolutions_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_fusionnet_model)

    log_training_settings(
        log_path,
        # Training settings
        batch_size=batch_size,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        loss_func=loss_func,
        w_smoothness=w_smoothness,
        w_weight_decay=w_weight_decay,
        w_lidar_loss=w_lidar_loss,
        loss_smoothness_kernel_size=loss_smoothness_kernel_size,
        outlier_removal_kernel_size=ground_truth_outlier_removal_kernel_size,
        outlier_removal_threshold=ground_truth_outlier_removal_threshold,
        ground_truth_dilation_kernel_size=ground_truth_dilation_kernel_size)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=checkpoint_dirpath,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        start_step_validation=start_step_validation,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    # Initialize optimizer with starting learning rate
    optimizer = torch.optim.Adam([
        {
            'params' : parameters_fusionnet_model,
            'weight_decay' : w_weight_decay
        }],
        lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        train_step, optimizer = fusionnet_model.restore_model(
            restore_path,
            optimizer=optimizer)

        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    time_start = time.time()

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for batch_data in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]

            # Unpack data
            image, depth, response, ground_truth, lidar_map = batch_data

            # Apply augmentations and data transforms
            [image], [depth, response, ground_truth, lidar_map] = train_transforms.transform(
                images_arr=[image],
                range_maps_arr=[depth, response, ground_truth, lidar_map],
                random_transform_probability=augmentation_probability)

            input_depth = torch.cat([depth, response], dim=1)

            # Forward through the network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth)

            # Compute loss function
            if ground_truth_dilation is not None:
                ground_truth = ground_truth_dilation(ground_truth)

            if ground_truth_outlier_removal is not None:
                ground_truth = ground_truth_outlier_removal.remove_outliers(ground_truth)
                
            validity_map_loss_smoothness = torch.where(
                ground_truth > 0,
                torch.zeros_like(ground_truth),
                torch.ones_like(ground_truth))

            loss, loss_info = fusionnet_model.compute_loss(
                image=image,
                output_depth=output_depth,
                ground_truth=ground_truth,
                lidar_map=lidar_map,
                loss_func=loss_func,
                w_smoothness=w_smoothness,
                loss_smoothness_kernel_size=loss_smoothness_kernel_size,
                validity_map_loss_smoothness=validity_map_loss_smoothness,
                w_lidar_loss=w_lidar_loss)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_per_summary) == 0:

                with torch.no_grad():
                    # Log tensorboard summary
                    fusionnet_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        image=image,
                        input_depth=depth,
                        input_response=response,
                        output_depth=output_depth,
                        ground_truth=ground_truth,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if train_step >= start_step_validation:
                    # Switch to validation mode
                    fusionnet_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=fusionnet_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=4,
                            log_path=log_path)

                    # Switch back to training
                    fusionnet_model.train()

                # Save checkpoints
                fusionnet_model.save_model(
                    depth_model_checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)

    # Evaluate once more after we are done training
    fusionnet_model.eval()

    with torch.no_grad():
        best_results = validate(
            model=fusionnet_model,
            dataloader=val_dataloader,
            transforms=val_transforms,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            n_summary_display=4,
            log_path=log_path)

    # Save checkpoints
    fusionnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

def validate(model,
             dataloader,
             transforms,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_summary_display=4,
             n_summary_display_interval=250,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    input_depth_summary = []
    response_summary = []
    ground_truth_summary = []

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, depth, response, ground_truth = inputs

        [image] = transforms.transform(
            images_arr=[image],
            random_transform_probability=0.0)

        input_depth = torch.cat([depth, response], dim=1)

        # Forward through network
        output_depth = model.forward(
            image=image,
            input_depth=input_depth)

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            input_depth_summary.append(depth)
            response_summary.append(response)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map = np.where(ground_truth > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None:
        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image=torch.cat(image_summary, dim=0),
            input_depth=torch.cat(input_depth_summary, dim=0),
            input_response=torch.cat(response_summary, dim=0),
            output_depth=torch.cat(output_depth_summary, dim=0),
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_display=n_summary_display)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        imae=imae,
        irmse=irmse,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        imae=best_results['imae'],
        irmse=best_results['irmse'],
        step=best_results['step'],
        log_path=log_path)

    return best_results

def run(restore_path,
        image_path,
        depth_path,
        response_path,
        ground_truth_path,
        # Input settings
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        # Network settings
        encoder_type,
        n_filters_encoder_image,
        n_filters_encoder_depth,
        fusion_type,
        decoder_type,
        n_filters_decoder,
        n_resolutions_decoder,
        min_predict_depth,
        max_predict_depth,
        # Weight settings
        weight_initializer,
        activation_func,
        # Output settings
        output_dirpath,
        save_outputs,
        keep_input_filenames,
        verbose=True,
        # Evaluation settings
        min_evaluate_depth=0.0,
        max_evaluate_depth=100.0):

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output directory
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    log_path = os.path.join(output_dirpath, 'results.txt')

    '''
    Set up paths for evaluation
    '''
    image_paths = data_utils.read_paths(image_path)
    depth_paths = data_utils.read_paths(depth_path)
    response_paths = data_utils.read_paths(response_path)

    n_sample = len(image_paths)

    ground_truth_available = \
        ground_truth_path is not None and \
        os.path.exists(ground_truth_path)

    if ground_truth_available:
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = [None] * n_sample

    for paths in [depth_paths, response_paths, ground_truth_paths]:
        assert n_sample == len(paths)

    dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetInferenceDataset(
            image_paths=image_paths,
            depth_paths=depth_paths,
            response_paths=response_paths,
            ground_truth_paths=ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up output paths
    '''
    if save_outputs:
        output_image_dirpath = os.path.join(output_dirpath, 'image')
        output_ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')
        output_depth_fusion_dirpath = os.path.join(output_dirpath, 'output_depth_fusion')
        output_depth_radar_dirpath = os.path.join(output_dirpath, 'output_depth_radar')
        output_response_radar_dirpath = os.path.join(output_dirpath, 'output_response_radar')

        output_dirpaths = [
            output_image_dirpath,
            output_ground_truth_dirpath,
            output_depth_fusion_dirpath,
            output_depth_radar_dirpath,
            output_response_radar_dirpath
        ]

        for dirpath in output_dirpaths:
            os.makedirs(dirpath, exist_ok=True)

    '''
    Build network and restore
    '''
    # Build network
    fusionnet_model = FusionNetModel(
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        decoder_type=decoder_type,
        n_resolution_decoder=n_resolutions_decoder,
        n_filters_decoder=n_filters_decoder,
        deconv_type='up',
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    fusionnet_model.eval()
    fusionnet_model.to(device)
    fusionnet_model.data_parallel()

    parameters_fusionnet_model = fusionnet_model.parameters()

    step, _ = fusionnet_model.restore_model(restore_path)

    '''
    Log settings
    '''
    log('Evaluation input paths:', log_path)
    input_paths = [
        image_path,
        depth_path,
        response_path
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=3,
        input_channels_depth=3,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolutions_decoder=n_resolutions_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_fusionnet_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=output_dirpath,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    if ground_truth_available:
        # Define evaluation metrics
        mae = np.zeros(n_sample)
        rmse = np.zeros(n_sample)
        imae = np.zeros(n_sample)
        irmse = np.zeros(n_sample)

    with torch.no_grad():

        for idx, data in enumerate(dataloader):
            # Move inputs to device
            data = [
                datum.to(device) for datum in data
            ]

            if ground_truth_available:
                image, depth, response, ground_truth = data
            else:
                image, depth, response = data

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            input_depth = torch.cat([depth, response], dim=1)

            # Forward through network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth)

            output_depth_fusion = np.squeeze(output_depth.cpu().numpy())

            if verbose:
                print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

            '''
            Evaluate results
            '''
            if ground_truth_available:
                # Convert to numpy to validate
                ground_truth = np.squeeze(ground_truth.cpu().numpy())

                validity_map = np.where(ground_truth > 0, 1, 0)

                # Select valid regions to evaluate
                validity_mask = np.where(validity_map > 0, 1, 0)
                min_max_mask = np.logical_and(
                    ground_truth > min_evaluate_depth,
                    ground_truth < max_evaluate_depth)
                mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

                # Compute validation metrics
                mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth_fusion[mask], 1000.0 * ground_truth[mask])
                rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_fusion[mask], 1000.0 * ground_truth[mask])
                imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_fusion[mask], 0.001 * ground_truth[mask])
                irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_fusion[mask], 0.001 * ground_truth[mask])

            '''
            Save outputs
            '''
            if save_outputs:

                if keep_input_filenames:
                    filename = os.path.splitext(os.path.basename(image_paths[idx]))[0] + '.png'
                else:
                    filename = '{:010d}.png'.format(idx)

                # Create output paths
                output_image_path = os.path.join(output_image_dirpath, filename)
                output_ground_truth_path = os.path.join(output_ground_truth_dirpath, filename)
                output_depth_fusion_path = os.path.join(output_depth_fusion_dirpath, filename)
                output_depth_radar_path = os.path.join(output_depth_radar_dirpath, filename)
                output_response_radar_path = os.path.join(output_response_radar_dirpath, filename)

                # Convert torch tensors to numpy (depth and ground truth already in numpy)
                output_image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
                output_depth_radar = np.squeeze(depth.cpu().numpy())
                output_response_radar = np.squeeze(response.cpu().numpy())

                # Save outputs
                output_image = (255 * output_image).astype(np.uint8)
                Image.fromarray(output_image).save(output_image_path)

                data_utils.save_depth(output_depth_fusion, output_depth_fusion_path)
                data_utils.save_depth(output_depth_radar, output_depth_radar_path)
                data_utils.save_response(output_response_radar, output_response_radar_path)

                if ground_truth_available:
                    data_utils.save_depth(ground_truth, output_ground_truth_path)

    '''
    Print evaluation results
    '''
    if ground_truth_available:
        # Compute mean metrics
        mae   = np.mean(mae)
        rmse  = np.mean(rmse)
        imae  = np.mean(imae)
        irmse = np.mean(irmse)

        # Print evaluation results to console
        log_evaluation_results(
            title='Evaluation results',
            mae=mae,
            rmse=rmse,
            imae=imae,
            irmse=irmse,
            step=step,
            log_path=log_path)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_channels_image,
                       input_channels_depth,
                       normalized_image_range):

    log('Input settings:', log_path)
    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         encoder_type,
                         n_filters_encoder_image,
                         n_filters_encoder_depth,
                         fusion_type,
                         decoder_type,
                         n_filters_decoder,
                         n_resolutions_decoder,
                         min_predict_depth,
                         max_predict_depth,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('encoder_type={}  fusion_type={}'.format(encoder_type, fusion_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_filters_encoder_depth={}'.format(n_filters_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('n_resolutions_decoder={}'.format(
        n_resolutions_decoder),
        log_path)
    log('min_predict_depth={}  max_predict_depth={}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          batch_size,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation,
                          augmentation_random_flip_type):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}  batch_size={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step, batch_size),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)

    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           loss_func,
                           w_smoothness,
                           w_weight_decay,
                           w_lidar_loss,
                           loss_smoothness_kernel_size,
                           outlier_removal_kernel_size,
                           outlier_removal_threshold,
                           ground_truth_dilation_kernel_size):

    log('Loss function settings:', log_path)
    log('loss_func={}'.format(
        loss_func),
        log_path)
    log('w_smoothness={:.1e}  w_weight_decay={:.1e}  w_lidar_loss={:.1e}'.format(
        w_smoothness, w_weight_decay, w_lidar_loss),
        log_path)
    log('loss_smoothness_kernel_size={}'.format(
        loss_smoothness_kernel_size),
        log_path)
    log('Ground truth preprocessing:')
    log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
        outlier_removal_kernel_size, outlier_removal_threshold),
        log_path)
    log('dilation_kernel_size={}'.format(
        ground_truth_dilation_kernel_size),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_dirpath=None,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        start_step_validation=None,
                        restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_dirpath is not None:
        log('checkpoint_path={}'.format(checkpoint_dirpath), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        if start_step_validation is not None:
            log('start_step_validation={}'.format(start_step_validation), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_path is not None and restore_path != '':
        log('restore_path={}'.format(restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)

def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           step=-1,
                           log_path=None):

    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse),
        log_path)
