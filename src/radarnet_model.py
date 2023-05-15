import torch, torchvision
import numpy as np
import log_utils
import networks


class RadarNetModel(object):
    '''
    Image radar fusion to determine correspondence of radar to image

    Arg(s):
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        input_patch_size_image : int
            patch of image to consider for radar point
        encoder_type : str
            encoder type
        n_filters_encoder_image : list[int]
            list of filters for each layer in image encoder
        n_neurons_encoder_image : list[int]
            list of neurons for each layer in depth encoder
        decoder_type : str
            decoder type
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 input_channels_image,
                 input_channels_depth,
                 input_patch_size_image,
                 encoder_type,
                 n_filters_encoder_image,
                 n_neurons_encoder_depth,
                 decoder_type,
                 n_filters_decoder,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 device=torch.device('cuda')):

        self.input_patch_size_image = input_patch_size_image
        self.device = device

        # height, width = input_patch_size_image
        # latent_height = np.ceil(height / 32.0).astype(int)
        # latent_width = np.ceil(width / 32.0).astype(int)

        height, width = input_patch_size_image
        latent_height = int((height // 32.0))
        latent_width = int((width // 32.0))

        latent_size_depth = latent_height * latent_width * n_neurons_encoder_depth[-1]

        # Build encoder
        if 'radarnetv1' in encoder_type:
            self.encoder = networks.RadarNetV1Encoder(
                input_channels_image=input_channels_image,
                input_channels_depth=input_channels_depth,
                input_patch_size_image=input_patch_size_image,
                n_filters_encoder_image=n_filters_encoder_image,
                n_neurons_encoder_depth=n_neurons_encoder_depth,
                latent_size_depth=latent_size_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)
        else:
            raise ValueError('Encoder type {} not supported.'.format(encoder_type))

        # Calculate number of channels for latent and skip connections combining image + depth
        n_skips = n_filters_encoder_image[:-1]
        n_skips = n_skips[::-1] + [0]

        latent_channels = n_filters_encoder_image[-1] + n_neurons_encoder_depth[-1]

        # Build decoder
        if 'multiscale' in decoder_type:
            self.decoder = networks.MultiScaleDecoder(
                input_channels=latent_channels,
                output_channels=1,
                n_resolution=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type='up')
        else:
            raise ValueError('Decoder type {} not supported.'.format(decoder_type))

        # Move to device
        self.to(self.device)

    def forward(self, image, point, bounding_boxes, return_logits=True):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            point : torch.Tensor[float32]
                N x 3 input point
            return_logits : bool
                if set, then return logits otherwise sigmoid
        Returns:
            torch.Tensor[float32] : N x 1 x H x W logits (correspondence map)
        '''

        latent, skips = self.encoder(image, point, bounding_boxes)

        logits = self.decoder(x=latent, skips=skips, shape=self.input_patch_size_image)[-1]

        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)

    def compute_loss(self,
                     logits,
                     ground_truth,
                     validity_map,
                     w_positive_class=1.0):
        '''
        Computes loss function

        Arg(s):
            logits : torch.Tensor[float32]
                N x 1 x H x W logits
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations to compute loss
            w_positive_class : float
                weight of positive class
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        device = logits.device

        # Define loss function
        w_positive_class = torch.tensor(w_positive_class, device=device)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits,
            target=ground_truth,
            reduction='none',
            pos_weight=w_positive_class)

        # Compute binary cross entropy
        loss = validity_map * loss
        loss = torch.sum(loss) / torch.sum(validity_map)

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}
        checkpoint['train_step'] = step
        checkpoint['radarnet_optimizer_state_dict'] = optimizer.state_dict()

        # Load weights for encoder, and decoder
        checkpoint['radarnet_encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['radarnet_decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore weights for encoder, and decoder
        self.encoder.load_state_dict(checkpoint['radarnet_encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['radarnet_decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['radarnet_optimizer_state_dict'])

        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image=None,
                    output_response=None,
                    output_label=None,
                    output_depth=None,
                    validity_map=None,
                    ground_truth_label=None,
                    ground_truth_depth=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32]
                N x 3 x H x W image
            output_response : torch.Tensor[float32]
                N x 1 x H x W soft correspondence map
            output_label : torch.Tensor[float32]
                N x 1 x H x W binary correspondence map
            output_depth : torch.Tensor[float32]
                N x 1 x H x W depth map
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map
            ground_truth_label : torch.Tensor[float32]
                N x 1 x H x W ground truth label
            ground_truth_depth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth map
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []

            display_summary_image_text = tag

            if image is not None:
                image_summary = image[0:n_display, ...]

                display_summary_image_text += '-image'

                # Add to list of images to log
                display_summary_image.append(image_summary.cpu())

            if output_response is not None:
                output_response_summary = output_response[0:n_display, ...]

                display_summary_image_text += '-output_response'

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        output_response_summary.cpu(),
                        colormap='inferno'))

                # Log distribution of output response
                summary_writer.add_histogram(
                    tag + '-output_response_distro',
                    output_response_summary,
                    global_step=step)

            if output_label is not None:
                output_label_summary = output_label[0:n_display, ...]

                display_summary_image_text += '-output_label'

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        output_label_summary.cpu(),
                        colormap='inferno'))

                # Log distribution of output and label
                summary_writer.add_histogram(
                    tag + '-output_label_distro',
                    output_label_summary,
                    global_step=step)

            if ground_truth_label is not None:
                ground_truth_label_summary = ground_truth_label[0:n_display, ...]

                validity_map_label_summary = torch.where(
                    ground_truth_label_summary > 0,
                    torch.ones_like(ground_truth_label_summary),
                    torch.zeros_like(ground_truth_label_summary))

                display_summary_image_text += '-ground_truth_label'

                if output_label is not None:
                    display_summary_image_text += '-error'

                    # Compute output error w.r.t. ground truth
                    ground_truth_label_error_summary = \
                        torch.abs(output_label_summary - ground_truth_label_summary)

                    ground_truth_label_error_summary = torch.where(
                        validity_map_label_summary == 1.0,
                        (ground_truth_label_error_summary + 1e-8) / (ground_truth_label_summary + 1e-8),
                        validity_map_label_summary)

                    display_summary_image.append(
                        log_utils.colorize(
                            ground_truth_label_error_summary.cpu(),
                            colormap='inferno'))

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        ground_truth_label_summary.cpu(),
                        colormap='inferno'))

                # Log distribution of ground truth
                summary_writer.add_histogram(
                    tag + '_ground_truth_label_distro',
                    ground_truth_label,
                    global_step=step)

            if validity_map is not None:
                validity_map_summary = validity_map[0:n_display, ...]

                display_summary_image_text += '-validity_map'

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        validity_map_summary.cpu(),
                        colormap='inferno'))

            if output_depth is not None:
                output_depth_summary = output_depth[0:n_display, ...]

                display_summary_image_text += '-output_depth'

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        (output_depth_summary / 100.0).cpu(),
                        colormap='viridis'))

                # Log distribution of output depth
                summary_writer.add_histogram(
                    tag + '-output_depth_distro',
                    output_depth,
                    global_step=step)

            if ground_truth_depth is not None:
                ground_truth_depth = torch.unsqueeze(ground_truth_depth[:, 0, :, :], dim=1)
                ground_truth_depth_summary = ground_truth_depth[0:n_display, ...]

                validity_map_summary = torch.where(
                    ground_truth_depth_summary > 0,
                    torch.ones_like(ground_truth_depth_summary),
                    torch.zeros_like(ground_truth_depth_summary))

                display_summary_image_text += '-ground_truth_label'

                if output_depth is not None:
                    display_summary_image_text += '-error'

                    # Compute output error w.r.t. ground truth
                    ground_truth_depth_error_summary = \
                        torch.abs(output_depth_summary - ground_truth_depth_summary)

                    ground_truth_depth_error_summary = torch.where(
                        validity_map_summary == 1.0,
                        (ground_truth_depth_error_summary + 1e-8) / (ground_truth_depth_summary + 1e-8),
                        validity_map_summary)

                    display_summary_image.append(
                        log_utils.colorize(
                            (ground_truth_depth_error_summary / 0.05).cpu(),
                            colormap='inferno'))

                # Add to list of images to log
                display_summary_image.append(
                    log_utils.colorize(
                        (ground_truth_depth_summary / 100.0).cpu(),
                        colormap='viridis'))

                # Log distribution of ground truth
                summary_writer.add_histogram(
                    tag + '-ground_truth_distro',
                    ground_truth_depth,
                    global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '-' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) > 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                    global_step=step)
