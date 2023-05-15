import os, time, warnings
import numpy as np

# Dependencies for network, loss, etc.
import torch, torchvision
import torchvision.transforms.functional as functional
import eval_utils, losses
from networks import RadarNet
from typing import NamedTuple

# Dependencies for data loading
from data_utils import Data_Utilities
from dataset import BinaryClassificationDataset, BinaryClassificationDatasetVal
from transforms import Transforms

# Dependencies for logging
import log_utils
from log_utils import log
from torch.utils.tensorboard import SummaryWriter

import pickle

warnings.filterwarnings("ignore")


def train(gt_train_paths,
          radar_train_paths,
          gt_val_paths,
          radar_val_paths,
          data_path,
          image_path,
          epsilon,
          # Input settings
          batch_size,
          patch_size,
          normalized_image_range,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_brightness,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          augmentation_random_flip_type,
          # Loss settings
          w_cross_entropy,
          w_smoothness,
          w_weight_decay,
          kernel_size_smoothness,
          set_invalid_to_negative,
          w_positive_class,
          # Checkpoint and summary settings
          checkpoint_dirpath,
          num_step_per_summary,
          num_step_per_checkpoint,
          start_step_validation,
          restore_path,
          # Hardware and debugging
          debug=False,
          num_workers=10):

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up checkpoint directory
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    checkpoint_path = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

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

    # Log input paths
    log('Input paths', log_path)
    log(gt_train_paths, log_path)
    log(gt_val_paths, log_path)
    log('', log_path)


    # path to the file that contains paths for pseudo GT training set
    open_file = open(gt_train_paths, "rb")
    gt_train_paths = pickle.load(open_file)
    open_file.close()

    # path to radar numpy arrays
    open_file = open(radar_train_paths, "rb")
    radar_train_paths = pickle.load(open_file)
    open_file.close()

    # path to the file that contains paths for pseudo GT training set
    open_file = open(gt_val_paths, "rb")
    gt_val_paths = pickle.load(open_file)
    open_file.close()

    # path to radar numpy arrays
    open_file = open(radar_val_paths, "rb")
    radar_val_paths = pickle.load(open_file)
    open_file.close()


    if debug:
        end_train_list = 100
        end_val_list = end_train_list + 50

        train_dataset = BinaryClassificationDataset(
            ground_truth_paths=gt_train_paths[0:end_train_list],
            radar_points_paths=radar_train_paths[0:end_train_list],
            image_dirpath=image_path,
            data_dirpath=data_path,
            patch_size=patch_size)
        val_dataset = BinaryClassificationDatasetVal(
            ground_truth_paths=gt_val_paths[end_train_list:end_val_list],
            radar_points_paths=radar_val_paths[end_train_list:end_val_list],
            image_dirpath=image_path,
            data_dirpath=data_path,
            patch_size=patch_size)
    else:
        train_dataset = BinaryClassificationDataset(
            ground_truth_paths=gt_train_paths,
            radar_points_paths=radar_train_paths,
            image_dirpath=image_path,
            data_dirpath=data_path,
            patch_size=patch_size)
        val_dataset = BinaryClassificationDatasetVal(
            ground_truth_paths=gt_val_paths,
            radar_points_paths=radar_val_paths,
            image_dirpath=image_path,
            data_dirpath=data_path,
            patch_size=patch_size)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        crop_image_to_shape_on_point=patch_size,
        random_brightness=augmentation_random_brightness,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_flip_type=augmentation_random_flip_type)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    num_train_sample = len(gt_train_paths)
    num_train_step = \
        learning_schedule[-1] * np.ceil(num_train_sample / batch_size).astype(np.int32)

    # Build network
    height, width = patch_size
    latent_height = np.ceil(height / 32.0).astype(int)
    latent_width = np.ceil(width / 32.0).astype(int)

    n_filters_encoder_image = [32, 64, 128, 128, 128]
    n_filters_encoder_depth = [32, 64, 128, 128, 128]
    latent_depth = n_filters_encoder_depth[-1]

    n_filters_decoder = [256, 128, 64, 32, 16]

    model = RadarNet(
        input_channels_image=3,
        input_channels_depth=3,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_output_depth=latent_height * latent_width * latent_depth,
        n_filters_decoder=n_filters_decoder,
        weight_initializer='kaiming_uniform',
        activation_func='leaky_relu',
        use_batch_norm=True)

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    parameters = model.parameters()

    # Log settings used during training
    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        num_train_sample, learning_schedule[-1], num_train_step),
        log_path)
    log('normalized_image_range={}'.format(
        normalized_image_range), log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (num_train_sample // batch_size), le * (num_train_sample // batch_size), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (num_train_sample // batch_size), le * (num_train_sample // batch_size), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_brightness={}'.format(
        augmentation_random_brightness), log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={:.1e}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread), log_path)
    log('augmentation_random_flip_type={}'.format(
        augmentation_random_flip_type), log_path)
    log('', log_path)

    log('Loss function settings:', log_path)
    log('w_scross_entropy={:.1e}  w_smoothness={:.1e}  w_weight_decay={:.1e}'.format(
        w_cross_entropy, w_smoothness, w_weight_decay),
        log_path)
    log('', log_path)

    log('Tensorboard settings:', log_path)
    log('event_path={}'.format(event_path), log_path)
    log('num_step_per_summary={}'.format(num_step_per_summary), log_path)
    log('', log_path)

    log('Checkpoint settings:', log_path)
    log('checkpoint_path={}'.format(checkpoint_dirpath), log_path)
    log('num_step_per_checkpoint={}'.format(num_step_per_checkpoint), log_path)
    log('start_step_validation={}'.format(start_step_validation), log_path)
    log('', log_path)

    log('restore_path={}'.format(restore_path),
        log_path)
    log('', log_path)

    log('w_positive_class={}'.format(w_positive_class), log_path)
    log('', log_path)

    if set_invalid_to_negative:
        log('Invalid is set to negative', log_path)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer = torch.optim.Adam([
        {
            'params' : parameters,
            'weight_decay' : w_weight_decay
        }],
        lr=learning_rate)

    # Define loss function
    w_positive_class = torch.tensor(w_positive_class, device=device)
    cross_entropy_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=w_positive_class)

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        model, train_step, optimizer = restore_model(model, restore_path, optimizer)

    model.train()

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

            batch_data = [
                data.to(device) for data in batch_data
            ]

            # Load image (B x 3 x H x W), radar point (B x 3), and ground truth (B x 1 x H x W)
            image, radar_point, ground_truth = batch_data

            pseudo_ground_truth = torch.where(
            torch.abs(ground_truth - radar_point[:,2].view(radar_point.shape[0],1,1,1)*torch.ones_like(ground_truth)) < epsilon,
            torch.ones_like(ground_truth), 
            torch.zeros_like(ground_truth))

            pseudo_ground_truth = torch.where(
                ground_truth > 0,
                pseudo_ground_truth,
                torch.full_like(pseudo_ground_truth, fill_value=2))

            # pseudo_ground_truth = pseudo_ground_truth.astype(np.int)

            # pseudo_ground_truth = np.expand_dims(pseudo_ground_truth, 0)

            ground_truth = pseudo_ground_truth

            [radar_point], [image], [ground_truth] = train_transforms.transform(
                points_arr=[radar_point],
                images_arr=[image],
                labels_arr=[ground_truth],
                random_transform_probability=augmentation_probability)

            # Create valid locations to compute loss
            if set_invalid_to_negative:

                # Any invalid locations are not correspondences
                ground_truth = torch.where(
                    ground_truth > 1,
                    torch.zeros_like(ground_truth),
                    ground_truth)
                # Every pixel will be valid
                validity_map = torch.ones_like(ground_truth)
            else:
                # Mask out invalid pixels in loss
                validity_map = torch.where(
                    ground_truth > 1,
                    torch.zeros_like(ground_truth),
                    torch.ones_like(ground_truth))

            # Forward through network
            logits = model(image, radar_point, torch.tensor(patch_size, device=device))

            # Compute binary cross entropy
            loss_cross_entropy = validity_map * cross_entropy_loss_func(logits, ground_truth.float())
            loss_cross_entropy = torch.sum(loss_cross_entropy) / torch.sum(validity_map)

            # Compute smoothness loss
            sigmoid = torch.sigmoid(logits)

            if w_smoothness == 0:
                loss_smoothness = torch.tensor(0, device=device)
            else:
                loss_smoothness = losses.sobel_smoothness_loss_func(
                    sigmoid,
                    image,
                    filter_size=[1, 1] + kernel_size_smoothness)

            loss = w_cross_entropy * loss_cross_entropy + w_smoothness * loss_smoothness

            # Backwards pass and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log summary
            if (train_step % num_step_per_summary) == 0:

                with torch.no_grad():
                    # Log scalars
                    train_summary_writer.add_scalar(
                        'loss',
                        loss,
                        global_step=train_step)
                    train_summary_writer.add_scalar(
                        'loss_cross_entropy',
                        loss_cross_entropy,
                        global_step=train_step)
                    train_summary_writer.add_scalar(
                        'loss_smoothness',
                        loss_smoothness,
                        global_step=train_step)

                    image_display = image[0:4, ...].cpu()
                    sigmoid_summary = sigmoid[0:4, ...].detach().clone()
                    ground_truth_summary = ground_truth[0:4, ...].detach().clone()
                    validity_map_summary = validity_map[0:4, ...].detach().clone()
                    output_summary = torch.where(
                        sigmoid_summary > 0.5,
                        torch.ones_like(sigmoid_summary),
                        torch.zeros_like(sigmoid_summary))
                    ground_truth_summary = validity_map_summary * ground_truth_summary
                    error_map_summary = validity_map_summary * torch.abs(output_summary - ground_truth_summary)

                    # Log images
                    sigmoid_display = log_utils.colorize(sigmoid_summary.cpu(), colormap='inferno')
                    output_display = log_utils.colorize(output_summary.cpu(), colormap='inferno')
                    ground_truth_display = log_utils.colorize(ground_truth_summary.cpu(), colormap='inferno')
                    error_map_display = log_utils.colorize(error_map_summary.cpu(), colormap='inferno')

                    display = torch.cat([
                        image_display,
                        sigmoid_display,
                        output_display,
                        ground_truth_display,
                        error_map_display],
                        dim=-1)

                    train_summary_writer.add_image(
                        'train_image_response_output_groundtruth_error',
                        torchvision.utils.make_grid(display, nrow=1),
                        global_step=train_step)

                    # Add histograms
                    train_summary_writer.add_histogram(
                        'sigmoid',
                        sigmoid,
                        global_step=train_step)

                    correspondence = torch.where(
                        ground_truth == 1, 
                        ground_truth, 
                        torch.zeros_like(ground_truth))

                    n_correspondence = torch.mean(torch.sum(correspondence.float(), dim=[1, 2, 3]))

                    train_summary_writer.add_scalar(
                        'average_correspondence_per_point',
                        n_correspondence,
                        global_step=train_step)

            # Log results and save checkpoints
            if (train_step % num_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (num_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{} Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, num_train_step, time_elapse, time_remain),
                    log_path)

                log('Loss={:.5f}  Cross Entropy={:.5f}  Smoothness={:.5f}'.format(
                    loss.item(), loss_cross_entropy.item(), loss_smoothness.item()),
                    log_path)

                if train_step >= start_step_validation:

                    model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=0.0,
                            max_evaluate_depth=100.0,
                            device=device,
                            summary_writer=val_summary_writer,
                            log_path=log_path)

                    # Switch back to training
                    model.train()

                # Save model to checkpoint
                save_model(model, checkpoint_path.format(train_step), train_step, optimizer)

    # Evaluate once more after we are done training
    model.eval()

    with torch.no_grad():
        best_results = validate(
            model=model,
            dataloader=val_dataloader,
            transforms=val_transforms,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=0.0,
            max_evaluate_depth=100.0,
            device=device,
            summary_writer=val_summary_writer,
            log_path=log_path)

    # Save model to checkpoint
    save_model(model, checkpoint_path.format(train_step), train_step, optimizer)

def validate(model,
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

    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    mae_intersection = np.zeros(n_sample)
    rmse_intersection = np.zeros(n_sample)
    imae_intersection = np.zeros(n_sample)
    irmse_intersection = np.zeros(n_sample)

    n_valid_points_output = np.zeros(n_sample)
    n_valid_points_ground_truth = np.zeros(n_sample)
    n_valid_points_intersection = np.zeros(n_sample)

    image_summaries = []
    output_summaries = []
    ground_truth_summaries = []
    response_summaries = []

    for sample_idx, batch_data in enumerate(dataloader):

        batch_data = [
            data.to(device) for data in batch_data
        ]

        image, radar_points, ground_truth, patch_size, pad_size = batch_data

        patch_size = patch_size.squeeze()
        pad_size = pad_size.squeeze()

        [radar_points], [image] = transforms.transform(
            points_arr=[radar_points],
            images_arr=[image],
            random_transform_probability=0.0)


        image = functional.pad(
            image,
            (pad_size, 0, pad_size, 0),
            padding_mode='edge',
            fill=0)
        image = torch.squeeze(image, 0)

        radar_points = torch.squeeze(radar_points, 0)

        output_tiles = []
        if radar_points.dim() == 2:
            radar_points = torch.unsqueeze(radar_points, 0)

        # Radar points have been repeated to N x P x 3, select the first to get N x 3
        radar_points = radar_points[:, 0, :]

        x_shifts = radar_points[:, 0].clone() + pad_size
        radar_points[:, 0] = pad_size

        # Crop the image
        image_crops = []

        height = image.shape[-2]
        crop_height = height - patch_size[0]

        for x in x_shifts:
            image_crop = image[:, crop_height:, int(x)-pad_size:int(x)+pad_size]
            image_crops.append(image_crop)

        # N x 3 x h x w image crops
        image_crops = torch.stack(image_crops, dim=0)

        output_crops = torch.sigmoid(
            model(image_crops.float(), radar_points.float(), patch_size))

        for output_crop, x in zip(output_crops, x_shifts):
            output = torch.zeros([1, image.shape[-2], image.shape[-1]] , device=device)

            # Thresholding
            output_crop = torch.where(output_crop < 0.5, torch.zeros_like(output_crop), output_crop)
            # Add crop to output
            output[:, crop_height:, int(x)-pad_size:int(x)+pad_size] = output_crop
            output_tiles.append(output)

        output_tiles = torch.cat(output_tiles, dim=0)
        output_tiles = output_tiles[:, :, pad_size:-pad_size]

        # Find the max response over all tiles
        response, output = torch.max(output_tiles, dim=0, keepdim=True)

        # Fill in the map based on z value of the points chosen
        for point_idx in range(radar_points.shape[0]):
            output = torch.where(
                output == point_idx,
                torch.full_like(output, fill_value=radar_points[point_idx, 2]),
                output)

        # Leave as 0s if we did not predict
        output = torch.where(
            torch.max(output_tiles, dim=0, keepdim=True)[0] == 0,
            torch.zeros_like(output),
            output)

        # Display summary
        if sample_idx % 500 == 0:
            image = image[:, :, pad_size:-pad_size]
            image_summary = torch.unsqueeze(image, 0)
            ground_truth_summary = torch.unsqueeze(ground_truth, 0)
            output_summary = torch.unsqueeze(output, 0)
            response_summary = torch.unsqueeze(response, 0)

            image_summaries.append(image_summary)
            output_summaries.append(output_summary)
            response_summaries.append(response_summary)
            ground_truth_summaries.append(ground_truth_summary)

        # Do evaluation against ground truth here
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        output = np.squeeze(output.cpu().numpy())

        # Validity map of output -> locations where output is valid
        validity_map_output = np.where(output > 0, 1, 0)
        validity_map_gt = np.where(ground_truth > 0, 1, 0)
        validity_map_intersection = validity_map_output*validity_map_gt

        n_valid_points_intersection[sample_idx] = np.sum(validity_map_intersection)
        n_valid_points_output[sample_idx] = np.sum(validity_map_output)
        n_valid_points_ground_truth[sample_idx] = np.sum(validity_map_gt)

        # Select valid regions to evaluate
        min_max_mask = np.logical_and(
            ground_truth > 0,
            ground_truth < 100)
        mask = np.where(np.logical_and(validity_map_gt, min_max_mask) > 0)
        mask_intersection = np.where(np.logical_and(validity_map_intersection, min_max_mask) > 0)

        output_depth_all = output[mask]
        ground_truth_all = ground_truth[mask]

        output_depth_intersection = output[mask_intersection]
        ground_truth_intersection = ground_truth[mask_intersection]

        # Compute validation metrics
        mae[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_all, 1000.0 * ground_truth_all)
        rmse[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_all, 1000.0 * ground_truth_all)
        imae[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_all, 0.001 * ground_truth_all)
        irmse[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_all, 0.001 * ground_truth_all)

        # Compute validation metrics for intersection
        mae_intersection[sample_idx] = eval_utils.mean_abs_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
        rmse_intersection[sample_idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_intersection, 1000.0 * ground_truth_intersection)
        imae_intersection[sample_idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)
        irmse_intersection[sample_idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_intersection, 0.001 * ground_truth_intersection)

    n_valid_points_output = np.mean(n_valid_points_output)
    n_valid_points_intersection = np.mean(n_valid_points_intersection)
    n_valid_points_ground_truth = np.mean(n_valid_points_ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Compute mean metrics for intersection
    mae_intersection = mae_intersection[~np.isnan(mae_intersection)]
    rmse_intersection = rmse_intersection[~np.isnan(rmse_intersection)]
    imae_intersection = imae_intersection[~np.isnan(imae_intersection)]
    irmse_intersection = irmse_intersection[~np.isnan(irmse_intersection)]

    mae_intersection = np.mean(mae_intersection)
    rmse_intersection = np.mean(rmse_intersection)
    imae_intersection = np.mean(imae_intersection)
    irmse_intersection = np.mean(irmse_intersection)

    # Write metrics to tensorboard
    summary_writer.add_scalar('mae', mae, global_step=step)
    summary_writer.add_scalar('rmse', rmse, global_step=step)
    summary_writer.add_scalar('imae', imae, global_step=step)
    summary_writer.add_scalar('irmse', mae, global_step=step)
    summary_writer.add_scalar('mae_intersection', mae_intersection, global_step=step)
    summary_writer.add_scalar('rmse_intersection', mae_intersection, global_step=step)
    summary_writer.add_scalar('imae_intersection', mae_intersection, global_step=step)
    summary_writer.add_scalar('irmse_intersection', mae_intersection, global_step=step)

    # Log images to tensorboard
    image_summary = torch.cat(image_summaries, dim=0)
    output_summary = torch.cat(output_summaries, dim=0)
    response_summary = torch.cat(response_summaries, dim=0)
    ground_truth_summary = torch.cat(ground_truth_summaries, dim=0)

    image_display = image_summary.cpu()

    response_display = log_utils.colorize(
        response_summary.cpu(), colormap='inferno')

    output_display = log_utils.colorize(
        (output_summary / 80.0).cpu(), colormap='viridis')
    ground_truth_display = log_utils.colorize(
        (ground_truth_summary / 80.0).cpu(), colormap='viridis')

    error_display = torch.where(
        ground_truth_summary > 0,
        torch.abs(output_summary - ground_truth_summary) / (ground_truth_summary + 1e-8),
        ground_truth_summary)
    error_display = log_utils.colorize(
        (error_display / 0.1).cpu(), colormap='inferno')

    display = torch.cat([
        image_display,
        response_display,
        output_display,
        ground_truth_display,
        error_display],
        dim=-1)

    summary_writer.add_image(
        'val_image_response_output_groundtruth_error',
            torchvision.utils.make_grid(display, nrow=1),
        global_step=step)

    # Print validation results to console
    log('Validation results (full):', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    log('Validation results (intersection):', log_path)
    log('{:>16}  {:>16}  {:>16}'.format(
        '# Ground truth', '# Output', '# Intersection'),
        log_path)
    log('{:>16.3f}  {:>16.3f}  {:>16.3f}'.format(
        n_valid_points_ground_truth, n_valid_points_output, n_valid_points_intersection),
        log_path)

    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        '', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        '', mae_intersection, rmse_intersection, imae_intersection, irmse_intersection),
        log_path)

    n_improve = 0
    if np.round(mae_intersection, 2) <= np.round(best_results['mae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(rmse_intersection, 2) <= np.round(best_results['rmse_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(imae_intersection, 2) <= np.round(best_results['imae_intersection'], 2):
        n_improve = n_improve + 1
    if np.round(irmse_intersection, 2) <= np.round(best_results['irmse_intersection'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse
        best_results['mae_intersection'] = mae_intersection
        best_results['rmse_intersection'] = rmse_intersection
        best_results['imae_intersection'] = imae_intersection
        best_results['irmse_intersection'] = irmse_intersection
        best_results['n_valid_points_output'] = n_valid_points_output
        best_results['n_valid_points_ground_truth'] = n_valid_points_ground_truth
        best_results['n_valid_points_intersection'] = n_valid_points_intersection

    log('Best results (full):', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    log('Best results (intersection):', log_path)
    log('{:>16}  {:>16}  {:>16}'.format(
        '# Ground truth', '# Output', '# Intersection'),
        log_path)
    log('{:>16.3f}  {:>16.3f}  {:>16.3f}'.format(
        best_results['n_valid_points_ground_truth'],
        best_results['n_valid_points_output'],
        best_results['n_valid_points_intersection']),
        log_path)

    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        '', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        '',
        best_results['mae_intersection'],
        best_results['rmse_intersection'],
        best_results['imae_intersection'],
        best_results['irmse_intersection']),
        log_path)

    return best_results

def save_model(model, checkpoint_path, step, optimizer):
    '''
    Save weights of the model to checkpoint path

    Arg(s):
        model : torch.nn.Module
            torch model
        checkpoint_path : str
            path to save checkpoint
        step : int
            current training step
        optimizer : torch.optim
            optimizer
    '''

    checkpoint = {}
    # Save training state
    checkpoint['train_step'] = step
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save encoder and decoder weights
    checkpoint['model_state_dict'] = model.state_dict()

    torch.save(checkpoint, checkpoint_path)

def restore_model(model, checkpoint_path, optimizer=None, device=torch.device('cuda')):
    '''
    Restore weights of the model

    Arg(s):
        model : torch.nn.Module
            torch model
        checkpoint_path : str
            path to checkpoint
        optimizer : torch.optim
            optimizer
    Returns:
        int : current step in optimization
        torch.optim : optimizer with restored state
    '''

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore sparse to dense pool, encoder and decoder weights
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass

    # Return the current step and optimizer
    return model, checkpoint['train_step'], optimizer
