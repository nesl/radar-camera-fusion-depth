import os, time
import numpy as np

# Dependencies for network, loss, etc.
import torch, torchvision
from radarnet_model import RadarNetModel

# Dependencies for data loading
import datasets, data_utils, eval_utils
from radarnet_transforms import Transforms

# Dependencies for logging
from log_utils import log
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def train(train_image_path,
          train_radar_path,
          train_ground_truth_path,
          val_image_path,
          val_radar_path,
          val_ground_truth_path,
          # Input settings
          batch_size,
          patch_size,
          total_points_sampled,
          sample_probability_of_lidar,
          normalized_image_range,
          # Network settings
          encoder_type,
          n_filters_encoder_image,
          n_neurons_encoder_depth,
          decoder_type,
          n_filters_decoder,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          augmentation_random_flip_type,
          # Loss settings
          w_weight_decay,
          w_positive_class,
          max_distance_correspondence,
          set_invalid_to_negative_class,
          # Checkpoint and summary settings
          checkpoint_dirpath,
          n_step_per_summary,
          n_step_per_checkpoint,
          start_step_validation,
          restore_path,
          # Evaluation settings
          min_evaluate_depth=0.0,
          max_evaluate_depth=100.0,
          # Hardware settings
          n_thread=10):

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up checkpoint directory
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    # Keep track of best results so far
    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty,
        'mae_intersection': np.infty,
        'rmse_intersection': np.infty,
        'imae_intersection': np.infty,
        'irmse_intersection': np.infty,
        'n_valid_points_output': np.infty,
        'n_valid_points_ground_truth': np.infty,
        'n_valid_points_intersection': np.infty
    }

    '''
    Set up paths for training
    '''
    train_image_paths = data_utils.read_paths(train_image_path)
    train_radar_paths = data_utils.read_paths(train_radar_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)

    n_train_sample = len(train_image_paths)

    assert n_train_sample == len(train_radar_paths)
    assert n_train_sample == len(train_ground_truth_paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        datasets.RadarNetTrainingDataset(
            image_paths=train_image_paths,
            radar_paths=train_radar_paths,
            ground_truth_paths=train_ground_truth_paths,
            patch_size=patch_size,
            total_points_sampled=total_points_sampled,
            sample_probability_of_lidar=sample_probability_of_lidar),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_thread)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_flip_type=augmentation_random_flip_type)

    '''
    Set up paths for validation
    '''
    val_image_paths = data_utils.read_paths(val_image_path)
    val_radar_paths = data_utils.read_paths(val_radar_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    n_val_sample = len(val_image_paths)

    assert n_val_sample == len(val_radar_paths)
    assert n_val_sample == len(val_ground_truth_paths)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.RadarNetInferenceDataset(
            image_paths=val_image_paths,
            radar_paths=val_radar_paths,
            ground_truth_paths=val_ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    # Build network
    radarnet_model = RadarNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    radarnet_model.to(device)
    radarnet_model.data_parallel()

    parameters_radarnet_model = radarnet_model.parameters()

    '''
    Log settings
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_path,
        train_radar_path,
        train_ground_truth_path,
    ]

    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_radar_path,
        val_ground_truth_path
    ]

    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_radarnet_model)

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
        augmentation_random_noise_type=augmentation_random_noise_type,
        augmentation_random_noise_spread=augmentation_random_noise_spread,
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_weight_decay=w_weight_decay,
        w_positive_class=w_positive_class,
        max_distance_correspondence=max_distance_correspondence,
        set_invalid_to_negative_class=set_invalid_to_negative_class)

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
            'params' : parameters_radarnet_model,
            'weight_decay' : w_weight_decay
        }],
        lr=learning_rate)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        train_step, optimizer = radarnet_model.restore_model(
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

            # Load image (N x 3 x H x W), radar point (N x 3), and ground truth (N x 1 x H x W)
            image, radar_point, bounding_boxes_list, ground_truth_depth = batch_data

            # Apply augmentations and data transforms
            [image], [ground_truth_depth], [radar_point], [bounding_boxes_list] = train_transforms.transform(
                images_arr=[image],
                labels_arr=[ground_truth_depth],
                points_arr=[radar_point],
                bounding_boxes_arr=[bounding_boxes_list],
                random_transform_probability=augmentation_probability)

            # print(ground_truth_depth.shape)
            radar_points_for_summary = radar_point.clone()
            radar_point = radar_point.view(radar_point.shape[0]*radar_point.shape[1], radar_point.shape[2])

            # for radar_depth_idx in range(0,radar_point.shape[1]):
            #     radar_depth = radar_point[:, radar_depth_idx, 2].view(radar_point.shape[0], 1, 1, 1, 1)

            radar_depth = radar_point[..., 2].view(radar_point.shape[0], 1, 1, 1)
            ground_truth_depth = ground_truth_depth.view(ground_truth_depth.shape[0]*ground_truth_depth.shape[1], ground_truth_depth.shape[2], ground_truth_depth.shape[3], ground_truth_depth.shape[4])

            '''
            Create ground truth labels and validity map
            '''

            distance_radar_ground_truth_depth = \
                torch.abs(ground_truth_depth - radar_depth * torch.ones_like(ground_truth_depth))

            # Correspondences are any point less than distance threshold
            ground_truth_label = torch.where(
                distance_radar_ground_truth_depth < max_distance_correspondence,
                torch.ones_like(ground_truth_depth),
                torch.zeros_like(ground_truth_depth))

            # Any missing empty points will be marked as invalid
            ground_truth_label = torch.where(
                ground_truth_depth > 0,
                ground_truth_label,
                torch.zeros_like(ground_truth_label))

            # Create valid locations to compute loss
            if set_invalid_to_negative_class:
                # Every pixel will be valid
                validity_map = torch.ones_like(ground_truth_depth)
            else:
                # Mask out invalid pixels in loss
                validity_map = torch.where(
                    ground_truth_depth <= 0,
                    torch.zeros_like(ground_truth_depth),
                    torch.ones_like(ground_truth_depth))

            '''
            Forward through network and compute loss
            '''

            bounding_boxes_list_new = []
            for bounding_box_batch_idx in range(0,bounding_boxes_list.shape[0]):
                bounding_boxes_list_new.append(bounding_boxes_list[bounding_box_batch_idx])

            logits = radarnet_model.forward(image, radar_point, bounding_boxes_list_new, return_logits=True)


            # Compute loss
            ground_truth_label = ground_truth_label.float()

            loss, loss_info = radarnet_model.compute_loss(
                logits=logits,
                ground_truth=ground_truth_label,
                validity_map=validity_map,
                w_positive_class=w_positive_class)

            # Backwards pass and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # input()

            # Log summary
            if (train_step % n_step_per_summary) == 0:

                image_crops = []
                pad_size_x = patch_size[1] // 2

                # bounding_boxes_list = bounding_boxes_list.view(-1,4)

                # Crop image and ground truth
                for batch_idx in range(0,bounding_boxes_list.shape[0]):
                    for bounding_box_idx in range(0,bounding_boxes_list.shape[1]):
                        start_x = int(bounding_boxes_list[batch_idx,bounding_box_idx,0].item())
                        end_x = int(bounding_boxes_list[batch_idx,bounding_box_idx,2].item())
                        start_y = int(bounding_boxes_list[batch_idx,bounding_box_idx,1].item())
                        end_y = int(bounding_boxes_list[batch_idx,bounding_box_idx,3].item())
                        # input()
                        image_cropped = image[batch_idx, :, start_y:end_y, start_x:end_x]
                        image_crops.append(image_cropped)

                image_cropped_for_summary = torch.stack(image_crops, dim=0)

                # ground_truth_depth_for_summary = ground_truth_depth.view(image_cropped_for_summary.shape[0], 
                #     total_points_sampled, image_cropped_for_summary.shape[2], 
                #     image_cropped_for_summary.shape[3])

                # ground_truth_label_for_summary = ground_truth_label.view(image_cropped_for_summary.shape[0], 
                #     total_points_sampled, image_cropped_for_summary.shape[2], 
                #     image_cropped_for_summary.shape[3])

                # ground_truth_depth_for_summary = ground_truth_depth_for_summary[:,0,...]
                # ground_truth_label_for_summary = ground_truth_label_for_summary[:,0,...]


                # ground_truth_depth_for_summary = torch.unsqueeze(ground_truth_depth_for_summary, dim=1)
                # ground_truth_label_for_summary = torch.unsqueeze(ground_truth_label_for_summary, dim=1)

                with torch.no_grad():
                    # Convert logits to response and label
                    response = torch.sigmoid(logits)
                    label = torch.where(
                        response > 0.50,
                        torch.ones_like(response),
                        torch.zeros_like(response))

                    n_ground_truth_label = \
                        torch.mean(torch.sum(ground_truth_label.float(), dim=[1, 2, 3]))
                    n_label = \
                        torch.mean(torch.sum(label.float(), dim=[1, 2, 3]))

                    loss_info['average_ground_truth_label_per_point'] = n_ground_truth_label
                    loss_info['average_predicted_label_per_point'] = n_label

                    # Log tensorboard summary
                    radarnet_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        image=image_cropped_for_summary,
                        output_response=response,
                        output_label=label,
                        validity_map=validity_map,
                        ground_truth_label=ground_truth_label,
                        ground_truth_depth=ground_truth_depth,
                        scalars=loss_info,
                        n_display=4)

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{} Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, time_elapse, time_remain),
                    log_path)

                log('Loss={:.5f}'.format(loss.item()), log_path)

                if train_step >= start_step_validation:

                    radarnet_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=radarnet_model,
                            patch_size=patch_size,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            log_path=log_path)

                    # Switch back to training
                    radarnet_model.train()

                # Save model to checkpoint
                radarnet_model.save_model(
                    checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)

    # Evaluate once more after we are done training
    radarnet_model.eval()

    with torch.no_grad():
        best_results = validate(
            model=radarnet_model,
            patch_size=patch_size,
            dataloader=val_dataloader,
            transforms=val_transforms,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            log_path=log_path)

    # Save model to checkpoint
    radarnet_model.save_model(
        checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

def forward(model, image, radar_points, bounding_boxes_list, device=torch.device('cuda')):

    # Determine crop size for possible radar correspondence
    patch_size = model.input_patch_size_image
    pad_size = patch_size[1] // 2

    image = torchvision.transforms.functional.pad(
        image,
        (pad_size, 0, pad_size, 0),
        padding_mode='edge')
    # image = torch.squeeze(image, 0)
    start_y = image.shape[-2] - patch_size[0]

    output_tiles = []
    if radar_points.dim() == 3:
        # Convert to 1 x N x 3 to N x 3
        radar_points = torch.squeeze(radar_points, dim=0)

    x_shifts = radar_points[:, 0].clone()

    height = image.shape[-2]
    crop_height = height - patch_size[0]

    output_crops = model.forward(
        image=image,
        point=radar_points,
        bounding_boxes=bounding_boxes_list,
        return_logits=False)

    for output_crop, x in zip(output_crops, x_shifts):
        output = torch.zeros([1, image.shape[-2], image.shape[-1]] , device=device)

        # Thresholding any response less than 0.5 to 0
        output_crop = torch.where(output_crop < 0.5, torch.zeros_like(output_crop), output_crop)
        # Add crop to output
        output[:, crop_height:, int(x)-pad_size:int(x)+pad_size] = output_crop
        output_tiles.append(output)

    output_tiles = torch.cat(output_tiles, dim=0)
    output_tiles = output_tiles[:, :, pad_size:-pad_size]

    # Find the max response over all tiles
    output_response, output = torch.max(output_tiles, dim=0, keepdim=True)

    # Fill in the map based on z value of the points chosen
    for point_idx in range(radar_points.shape[0]):
        output = torch.where(
            output == point_idx,
            torch.full_like(output, fill_value=radar_points[point_idx, 2]),
            output)

    # Leave as 0s if we did not predict
    output_depth = torch.where(
        torch.max(output_tiles, dim=0, keepdim=True)[0] == 0,
        torch.zeros_like(output),
        output)

    return output_depth, output_response

def validate(model,
             patch_size,
             dataloader,
             transforms,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             log_path):

    n_sample = len(dataloader)

    # Define evaluation metrics
    mae_intersection = np.zeros(n_sample)
    rmse_intersection = np.zeros(n_sample)
    imae_intersection = np.zeros(n_sample)
    irmse_intersection = np.zeros(n_sample)

    n_valid_points_output = np.zeros(n_sample)
    n_valid_points_ground_truth = np.zeros(n_sample)
    n_valid_points_intersection = np.zeros(n_sample)

    image_summaries = []
    output_depth_summaries = []
    ground_truth_summaries = []

    for sample_idx, batch_data in enumerate(dataloader):

        batch_data = [
            data.to(device) for data in batch_data
        ]

        # 1 x 3 x H x W image, 1 x N x 3 points
        image, radar_points, ground_truth = batch_data

        bounding_boxes_list = []

        pad_size_x = patch_size[1] // 2
        radar_points = radar_points.squeeze(dim=0)

        if radar_points.ndim == 1:
            # Expand to 1 x 3
            radar_points = np.expand_dims(radar_points, axis=0)

        # get the shifted radar points after padding
        for radar_point_idx in range(0,radar_points.shape[0]):
            # Set radar point to the center of the patch
            radar_points[radar_point_idx,0] = radar_points[radar_point_idx,0] + pad_size_x
            bounding_box = torch.zeros(4)
            bounding_box[0] = radar_points[radar_point_idx,0] - pad_size_x
            bounding_box[1] = 0
            bounding_box[2] = radar_points[radar_point_idx,0] + pad_size_x
            bounding_box[3] = image.shape[-2]
            bounding_boxes_list.append(bounding_box)
            
        bounding_boxes_list = [torch.stack(bounding_boxes_list, dim=0)]


        [image], [radar_points], [bounding_boxes_list] = transforms.transform(
            images_arr=[image],
            points_arr=[radar_points],
            bounding_boxes_arr=[bounding_boxes_list],
            random_transform_probability=0.0)

        output_depth, output_response = forward(
            model=model,
            image=image,
            radar_points=radar_points,
            bounding_boxes_list=bounding_boxes_list,
            device=device)

        # Display summary
        if sample_idx % 500 == 0:
            image_summary = image
            ground_truth_summary = ground_truth
            output_depth_summary = torch.unsqueeze(output_depth, dim=0)

            image_summaries.append(image_summary)
            output_depth_summaries.append(output_depth_summary)
            ground_truth_summaries.append(ground_truth_summary)

        # Do evaluation against ground truth here
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        output_depth = np.squeeze(output_depth.cpu().numpy())

        # Validity map of output -> locations where output is valid
        validity_map_output = np.where(output_depth > 0, 1, 0)
        validity_map_ground_truth = np.where(ground_truth > 0, 1, 0)
        validity_map_intersection = validity_map_output * validity_map_ground_truth

        n_valid_points_intersection[sample_idx] = np.sum(validity_map_intersection)
        n_valid_points_output[sample_idx] = np.sum(validity_map_output)
        n_valid_points_ground_truth[sample_idx] = np.sum(validity_map_ground_truth)

        # Select valid regions to evaluate
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

        output_depth_intersection = output_depth[mask_intersection]
        ground_truth_intersection = ground_truth[mask_intersection]

        # Compute validation metrics for intersection
        mae_intersection[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
        rmse_intersection[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
        imae_intersection[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)
        irmse_intersection[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)

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

    # Log to tensorboard
    if summary_writer is not None:
        scalars = {
            'mae_intersection' : mae_intersection,
            'rmse_intersection' : rmse_intersection,
            'imae_intersection' : imae_intersection,
            'irmse_intersection' : irmse_intersection,
            'n_valid_points_output' : n_valid_points_output,
            'n_valid_points_intersection' : n_valid_points_intersection
        }

        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image=torch.cat(image_summaries, dim=0),
            output_depth=torch.cat(output_depth_summaries, dim=0),
            ground_truth_depth=torch.cat(ground_truth_summaries, dim=0),
            scalars=scalars,
            n_display=4)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae_intersection=mae_intersection,
        rmse_intersection=rmse_intersection,
        imae_intersection=imae_intersection,
        irmse_intersection=irmse_intersection,
        n_valid_points_output=n_valid_points_output,
        n_valid_points_intersection=n_valid_points_intersection,
        n_valid_points_ground_truth=n_valid_points_ground_truth,
        step=step,
        log_path=log_path)

    n_improve = 0
    if np.round(mae_intersection, 2) <= np.round(best_results['mae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(rmse_intersection, 2) <= np.round(best_results['rmse_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(imae_intersection, 2) <= np.round(best_results['imae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(irmse_intersection, 2) <= np.round(best_results['irmse_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(n_valid_points_intersection, 2) >= np.round(best_results['n_valid_points_intersection'], 2):
        n_improve = n_improve + 1

    if n_improve > 3:
        best_results['step'] = step
        best_results['mae_intersection'] = mae_intersection
        best_results['rmse_intersection'] = rmse_intersection
        best_results['imae_intersection'] = imae_intersection
        best_results['irmse_intersection'] = irmse_intersection
        best_results['n_valid_points_output'] = n_valid_points_output
        best_results['n_valid_points_ground_truth'] = n_valid_points_ground_truth
        best_results['n_valid_points_intersection'] = n_valid_points_intersection

    log_evaluation_results(
        title='Best results',
        mae_intersection=best_results['mae_intersection'],
        rmse_intersection=best_results['rmse_intersection'],
        imae_intersection=best_results['imae_intersection'],
        irmse_intersection=best_results['irmse_intersection'],
        n_valid_points_output=best_results['n_valid_points_output'],
        n_valid_points_intersection=best_results['n_valid_points_intersection'],
        n_valid_points_ground_truth=best_results['n_valid_points_ground_truth'],
        step=best_results['step'],
        log_path=log_path)

    return best_results

def run(restore_path,
        image_path,
        radar_path,
        ground_truth_path,
        # Input settings
        patch_size,
        normalized_image_range,
        # Network settings
        encoder_type,
        n_filters_encoder_image,
        n_neurons_encoder_depth,
        decoder_type,
        n_filters_decoder,
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
    radar_paths = data_utils.read_paths(radar_path)

    n_sample = len(image_paths)

    ground_truth_available = \
        ground_truth_path is not None and \
        os.path.exists(ground_truth_path)

    if ground_truth_available:
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = [None] * n_sample

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
        normalized_image_range=normalized_image_range)

    '''
    Set up output paths
    '''
    if save_outputs:
        output_image_dirpath = os.path.join(output_dirpath, 'image')
        output_ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_depth')
        output_response_dirpath = os.path.join(output_dirpath, 'output_response')

        output_dirpaths = [
            output_image_dirpath,
            output_ground_truth_dirpath,
            output_depth_dirpath,
            output_response_dirpath
        ]

        for dirpath in output_dirpaths:
            os.makedirs(dirpath, exist_ok=True)

    '''
    Build network and restore
    '''
    radarnet_model = RadarNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=patch_size,
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    radarnet_model.eval()
    radarnet_model.to(device)
    radarnet_model.data_parallel()

    parameters_radarnet_model = radarnet_model.parameters()

    radarnet_model.restore_model(restore_path)

    '''
    Log settings
    '''
    log('Evaluation input paths:', log_path)
    input_paths = [
        image_path,
        radar_path
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
        input_patch_size_image=patch_size,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_neurons_encoder_depth=n_neurons_encoder_depth,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_radarnet_model)

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
        mae_intersection = np.zeros(n_sample)
        rmse_intersection = np.zeros(n_sample)
        imae_intersection = np.zeros(n_sample)
        irmse_intersection = np.zeros(n_sample)

        n_valid_points_output = np.zeros(n_sample)
        n_valid_points_ground_truth = np.zeros(n_sample)
        n_valid_points_intersection = np.zeros(n_sample)

    with torch.no_grad():

        for idx, data in enumerate(dataloader):
            # Move inputs to device
            data = [
                datum.to(device) for datum in data
            ]

            # 1 x 3 x H x W image, 1 x N x 3 points
            if ground_truth_available:
                image, radar_points, ground_truth = data
            else:
                image, radar_points = data

            [image], [radar_points] = transforms.transform(
                images_arr=[image],
                points_arr=[radar_points],
                random_transform_probability=0.0)

            output_depth, output_response = forward(
                model=radarnet_model,
                image=image,
                radar_points=radar_points,
                device=device)

            output_depth = np.squeeze(output_depth.cpu().numpy())

            if verbose:
                print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

            '''
            Evaluate results
            '''
            if ground_truth_available:
                ground_truth = np.squeeze(ground_truth.cpu().numpy())

                # Validity map of output -> locations where output is valid
                validity_map_output = np.where(output_depth > 0, 1, 0)
                validity_map_ground_truth = np.where(ground_truth > 0, 1, 0)
                validity_map_intersection = validity_map_output * validity_map_ground_truth

                n_valid_points_intersection[idx] = np.sum(validity_map_intersection)
                n_valid_points_output[idx] = np.sum(validity_map_output)
                n_valid_points_ground_truth[idx] = np.sum(validity_map_ground_truth)

                # Select valid regions to evaluate
                min_max_mask = np.logical_and(
                    ground_truth > min_evaluate_depth,
                    ground_truth < max_evaluate_depth)
                mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

                output_depth_intersection = output_depth[mask_intersection]
                ground_truth_intersection = ground_truth[mask_intersection]

                # Compute validation metrics for intersection
                mae_intersection[idx] = eval_utils.mean_abs_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
                rmse_intersection[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
                imae_intersection[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)
                irmse_intersection[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)

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

                output_depth_path = os.path.join(output_depth_dirpath, filename)
                output_response_path = os.path.join(output_response_dirpath, filename)

                # Convert torch tensors to numpy (depth and ground truth already in numpy)
                output_image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
                output_response = np.squeeze(output_response.cpu().numpy())

                # Save outputs
                output_image = (255 * output_image).astype(np.uint8)
                Image.fromarray(output_image).save(output_image_path)

                data_utils.save_depth(output_depth, output_depth_path)
                data_utils.save_response(output_response, output_response_path)

                if ground_truth_available:
                    data_utils.save_depth(ground_truth, output_ground_truth_path)

    '''
    Print evaluation results
    '''
    if ground_truth_available:
        # Compute mean metrics for intersection
        n_valid_points_output = np.mean(n_valid_points_output)
        n_valid_points_intersection = np.mean(n_valid_points_intersection)
        n_valid_points_ground_truth = np.mean(n_valid_points_ground_truth)

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
            title='Evaluation results',
            mae_intersection=mae_intersection,
            rmse_intersection=rmse_intersection,
            imae_intersection=imae_intersection,
            irmse_intersection=irmse_intersection,
            n_valid_points_output=n_valid_points_output,
            n_valid_points_intersection=n_valid_points_intersection,
            n_valid_points_ground_truth=n_valid_points_ground_truth,
            step=-1,
            log_path=log_path)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_channels_image,
                       input_channels_depth,
                       input_patch_size_image,
                       normalized_image_range):

    log('Input settings:', log_path)
    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('input_patch_size_image={}'.format(
        input_patch_size_image),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         encoder_type,
                         n_filters_encoder_image,
                         n_neurons_encoder_depth,
                         decoder_type,
                         n_filters_decoder,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_neurons_encoder_depth={}'.format(n_neurons_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
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
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
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
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)

    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_weight_decay,
                           w_positive_class,
                           max_distance_correspondence,
                           set_invalid_to_negative_class):

    log('Loss function settings:', log_path)
    log('w_positve_class={:.1e}  w_weight_decay={:.1e}'.format(
        w_positive_class, w_weight_decay),
        log_path)
    log('max_distance_correspondence={}  set_invalid_to_negative_class={}'.format(
        max_distance_correspondence, set_invalid_to_negative_class),
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
                           mae_intersection,
                           rmse_intersection,
                           imae_intersection,
                           irmse_intersection,
                           n_valid_points_output,
                           n_valid_points_intersection,
                           n_valid_points_ground_truth,
                           step=-1,
                           log_path=None):

    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>14}  {:>14}  {:>14}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', '# Output', '# Intersection', '# Ground truth'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:14.3f}  {:14.3f}  {:14.3f}'.format(
        step,
        mae_intersection,
        rmse_intersection,
        imae_intersection,
        irmse_intersection,
        n_valid_points_output,
        n_valid_points_intersection,
        n_valid_points_ground_truth),
        log_path)
