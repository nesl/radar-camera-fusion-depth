import torch, torchvision
import log_utils
import fusionnet_losses as losses
import networks
import numpy as np

class FusionNetModel(object):
    '''
    Image radar fusion

    Arg(s):
        encoder_type : str
            encoder type
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        n_filters_encoder_image : list[int]
            list of filters for each layer in encoder
        n_filters_encoder_image : list[int]
            list of filters for each layer in encoder
        decoder_type : str
            decoder type
        n_resolution_decoder : int
            minimum resolution of multiscale outputs is 1/(2^n_resolution_decoder)
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        resolutions_depthwise_separable_decoder : list[int]
            resolutions to use depthwise separable convolution
        output_func: str
            output function of decoder
        activation_func : str
            activation function for network
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        use_batch_norm : bool
            if set, then applied batch normalization
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 input_channels_image,
                 input_channels_depth,
                 encoder_type,
                 n_filters_encoder_image,
                 n_filters_encoder_depth,
                 fusion_type,
                 decoder_type,
                 n_resolution_decoder,
                 n_filters_decoder,
                 deconv_type,
                 activation_func,
                 weight_initializer,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.encoder_type = encoder_type
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device

        # Calculate number of channels in encoder combined for skip connections
        if fusion_type == 'add' or fusion_type == 'weight':
            n_filters_encoder = n_filters_encoder_image
            latent_channels = n_filters_encoder[-1]
        elif fusion_type == 'weight_and_project':
            n_filters_encoder = n_filters_encoder_image
            latent_channels = n_filters_encoder[-1]
        elif fusion_type == 'concat':
            n_filters_encoder = [
                i + z
                for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
            ]
            latent_channels = n_filters_encoder[-1]
        else:
            raise ValueError('Unsupported fusion type: {}'.format(fusion_type))

        # Build encoder
        if 'fusionnet18' in encoder_type or 'resnet18' in encoder_type:
            n_layer = 18
        elif 'fusionnet34' in encoder_type or 'resnet34' in encoder_type:
            n_layer = 34
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        if 'fusionnet18' in encoder_type or 'fusionnet34' in encoder_type:
            self.encoder = networks.FusionNetEncoder(
                n_layer=n_layer,
                input_channels_image=input_channels_image,
                input_channels_depth=input_channels_depth,
                n_filters_encoder_image=n_filters_encoder_image,
                n_filters_encoder_depth=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type,
                fusion_type=fusion_type)
        elif 'resnet18' in encoder_type or 'resnet34' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=n_layer,
                input_channels=input_channels_image,
                n_filters=n_filters_encoder_image,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)

            n_filters_encoder = n_filters_encoder_image
            latent_channels = n_filters_encoder[-1]
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        # Calculate number of channels for latent and skip connections combining image + depth
        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        # Build decoder
        if 'multiscale' in decoder_type:
            self.decoder = networks.MultiScaleDecoder(
                input_channels=latent_channels,
                output_channels=1,
                n_resolution=n_resolution_decoder,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type=deconv_type)
        else:
            raise ValueError('Unsuported decoder type: {}'.format(decoder_type))

        # Move to device
        self.to(self.device)

    def forward(self, image, input_depth, return_multiscale=False):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            input_depth : torch.Tensor[float32]
                N x 1 x H x W input depth
            return_multiscale : bool
                if set, then return multiple outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        if 'resnet18' in self.encoder_type or 'resnet34' in self.encoder_type:
            latent, skips = self.encoder(x=image)
        else:
            latent, skips = self.encoder(image=image, depth=input_depth)

        outputs = self.decoder(x=latent, skips=skips, shape=image.shape[-2:])

        outputs = [
            self.min_predict_depth / (torch.sigmoid(out) + self.min_predict_depth / self.max_predict_depth)
            for out in outputs
        ]

        if return_multiscale:
            return outputs
        else:
            return outputs[-1]

    def compute_loss(self,
                     image,
                     output_depth,
                     ground_truth,
                     lidar_map,
                     loss_func,
                     w_smoothness,
                     loss_smoothness_kernel_size,
                     validity_map_loss_smoothness,
                     w_lidar_loss):
        '''
        Computes loss function

        Arg(s):
            image :  torch.Tensor[float32]
                N x 3 x H x W image
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth
            lidar_map : torch.Tensor[float32]
                N x 1 x H x W single lidar scan
            loss_func : str
                loss function to minimize
            w_smoothness : float
                weight of local smoothness loss
            loss_smoothness_kernel_size : tuple[int]
                kernel size of loss smoothness
            validity_map_loss_smoothness : torch.Tensor[float32]
                N x 1 x H x W validity map
            w_lidar_loss : float
                weight of lidar loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        loss = 0.0
        loss_supervised = 0.0
        loss_smoothness = 0.0
        loss_lidar = 0.0

        if w_lidar_loss > 0.0:
            # Mask out ground truth where lidar is available to avoid double counting
            mask_lidar = torch.where(
                lidar_map > 0.0,
                torch.zeros_like(lidar_map),
                torch.ones_like(lidar_map))

            ground_truth = ground_truth * mask_lidar

        # Get valid locations for supervision
        validity_map_ground_truth = ground_truth > 0
        validity_map_lidar = lidar_map > 0

        if not isinstance(output_depth, list):
            output_depth = [output_depth]

        for scale, output in enumerate(output_depth):

            output_height, output_width = output.shape[-2:]
            target_height, target_width = ground_truth.shape[-2:]

            if output_height > target_height and output_width > target_width:

                output = torch.nn.functional.interpolate(
                    output,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=True)

            w_scale = 1.0 / (2 ** (len(output_depth) - scale - 1))

            if loss_func == 'l1':
                loss_supervised = loss_supervised + w_scale * losses.l1_loss(
                    output[validity_map_ground_truth],
                    ground_truth[validity_map_ground_truth])

                if w_lidar_loss > 0.0:
                    loss_lidar = loss_lidar + w_scale * losses.l1_loss(
                        output[validity_map_lidar],
                        lidar_map[validity_map_lidar])

            elif loss_func == 'l2':
                loss_supervised = loss_supervised + w_scale * losses.l2_loss(
                    output[validity_map_ground_truth],
                    ground_truth[validity_map_ground_truth])

                if w_lidar_loss > 0.0:
                    loss_lidar = loss_lidar + w_scale * losses.l2_loss(
                        output[validity_map_lidar],
                        lidar_map[validity_map_lidar])

            elif loss_func == 'smoothl1':
                loss_supervised = loss_supervised + w_scale * losses.smooth_l1_loss(
                    output[validity_map_ground_truth],
                    ground_truth[validity_map_ground_truth])

                if w_lidar_loss > 0.0:
                    loss_lidar = loss_lidar + w_scale * losses.smooth_l1_loss(
                        output[validity_map_lidar],
                        lidar_map[validity_map_lidar])
            else:
                raise ValueError('No such loss: {}'.format(loss_func))

            if w_smoothness > 0.0:

                if loss_smoothness_kernel_size <= 1:
                    loss_smoothness = loss_smoothness + w_scale * losses.smoothness_loss_func(
                        image=image,
                        predict=output)
                else:
                    loss_smoothness_kernel_size = \
                        [1, 1, loss_smoothness_kernel_size, loss_smoothness_kernel_size]

                    loss_smoothness = loss_smoothness + w_scale * losses.sobel_smoothness_loss_func(
                        image=image,
                        predict=output,
                        weights=validity_map_loss_smoothness,
                        filter_size=loss_smoothness_kernel_size)

        loss = loss_supervised + w_smoothness * loss_smoothness + w_lidar_loss * loss_lidar

        loss_info = {
            'loss' : loss,
            'loss_supervised' : loss_supervised,
            'loss_smoothness' : loss_smoothness,
            'loss_lidar' : loss_lidar
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
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
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Load weights for encoder, and decoder
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()

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
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
                    input_depth=None,
                    input_response=None,
                    output_depth=None,
                    ground_truth=None,
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
            input_depth : torch.Tensor[float32]
                N x 1 x H x W input depth
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image is not None:
                image_summary = image[0:n_display, ...]

                display_summary_image_text += '_image'
                display_summary_depth_text += '_image'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image_summary.cpu(),
                        torch.zeros_like(image_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if output_depth is not None:
                output_depth_summary = output_depth[0:n_display, ...]

                display_summary_depth_text += '_output_depth'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

            if output_depth is not None and input_depth is not None:
                input_depth_summary = input_depth[0:n_display, ...]

                display_summary_depth_text += '_input_depth-error'

                # Compute output error w.r.t. input depth
                input_depth_error_summary = \
                    torch.abs(output_depth_summary - input_depth_summary)

                input_depth_error_summary = torch.where(
                    input_depth_summary > 0.0,
                    input_depth_error_summary / (input_depth_summary + 1e-8),
                    input_depth_summary)

                # Add to list of images to log
                input_depth_summary = log_utils.colorize(
                    (input_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                input_depth_error_summary = log_utils.colorize(
                    (input_depth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        input_depth_summary,
                        input_depth_error_summary],
                        dim=3))

                # Log distribution of input depth
                summary_writer.add_histogram(tag + '_input_depth_distro', input_depth, global_step=step)

            if output_depth is not None and input_response is not None:
                response_summary = input_response[0:n_display, ...]

                display_summary_depth_text += '_response'

                # Add to list of images to log
                response_summary = log_utils.colorize(
                    response_summary.cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        response_summary,
                        torch.zeros_like(response_summary)],
                        dim=3))

                # Log distribution of input depth
                summary_writer.add_histogram(tag + '_response_distro', input_depth, global_step=step)

            if output_depth is not None and ground_truth is not None:
                ground_truth = ground_truth[0:n_display, ...]
                ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

                ground_truth_summary = ground_truth[0:n_display]
                validity_map_summary = torch.where(
                    ground_truth > 0,
                    torch.ones_like(ground_truth),
                    torch.zeros_like(ground_truth))

                display_summary_depth_text += '_ground_truth-error'

                # Compute output error w.r.t. ground truth
                ground_truth_error_summary = \
                    torch.abs(output_depth_summary - ground_truth_summary)

                ground_truth_error_summary = torch.where(
                    validity_map_summary == 1.0,
                    (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                    validity_map_summary)

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth_error_summary = log_utils.colorize(
                    (ground_truth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth_summary,
                        ground_truth_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) > 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                    global_step=step)

            if len(display_summary_depth) > 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                    global_step=step)
