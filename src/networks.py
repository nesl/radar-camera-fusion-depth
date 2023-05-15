import torch
import net_utils
import torchvision

'''
Encoders
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks2 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks3 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks4 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks5 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks6 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks7 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks7 = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels,
                    out_channels,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels : int
                number of channels
            out_channels : int
                number of output channels
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels = out_channels
                stride = 1

            block = network_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks.append(block)

        blocks = torch.nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        '''
        Forward input x through the ResNet model
        Arg(s):
            x : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]

class FusionNetEncoder(torch.nn.Module):
    '''
    FusionNet encoder with skip connections
    Arg(s):
        n_layer : int
            number of layer for encoder
        input_channels_image : int
            number of channels in input data
        input_channels_depth : int
            number of channels in input data
        n_filters_per_block : list[int]
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        fusion_type : str
            add, weight
    '''

    def __init__(self,
                 n_layer=18,
                 input_channels_image=3,
                 input_channels_depth=3,
                 n_filters_encoder_image=[32, 64, 128, 256, 256],
                 n_filters_encoder_depth=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 fusion_type='add'):
        super(FusionNetEncoder, self).__init__()

        self.fusion_type = fusion_type

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        resnet_block = net_utils.ResNetBlock

        assert len(n_filters_encoder_image) == len(n_filters_encoder_depth)

        for n in range(len(n_filters_encoder_image) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters_encoder_image)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        self.conv1_image = net_utils.Conv2d(
            input_channels_image,
            n_filters_encoder_image[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv1_depth = net_utils.Conv2d(
            input_channels_depth,
            n_filters_encoder_depth[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv1_project = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv1_weight = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_depth[filter_idx],
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv1_weight = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv1_project = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks2_image, self.blocks2_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv2_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv2_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv2_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv2_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks3_image, self.blocks3_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv3_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv3_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv3_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv3_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks4_image, self.blocks4_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv4_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv4_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv4_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv4_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks5_image, self.blocks5_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv5_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv5_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        if fusion_type == 'weight_and_project':

            self.conv5_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv5_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]

            self.blocks6_image, self.blocks6_depth = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            if fusion_type == 'add':
                self.conv6_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)

            if fusion_type == 'weight_and_project':

                self.conv6_weight = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)

                self.conv6_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)
        else:
            self.blocks6_image = None
            self.blocks6_depth = None
            self.conv6_weight = None
            self.conv6_project = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]

            self.blocks7_image, self.blocks7_depth = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            if fusion_type == 'weight_and_project':

                self.conv7_weight = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)

                self.conv7_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)
        else:
            self.blocks7_image = None
            self.blocks7_depth = None
            self.conv7_weight = None
            self.conv7_project = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels_image,
                    in_channels_depth,
                    out_channels_image,
                    out_channels_depth,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels_image : int
                number of channels in image branch
            in_channels_depth : int
                number of channels in depth branch
            out_channels_image : int
                number of output channels in image branch
            out_channels_depth : int
                number of output channels in depth branch
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks_image = []
        blocks_depth = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels_image = out_channels_image
                in_channels_depth = out_channels_depth
                stride = 1

            block_image = network_block(
                in_channels=in_channels_image,
                out_channels=out_channels_image,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_image.append(block_image)

            block_depth = network_block(
                in_channels=in_channels_depth,
                out_channels=out_channels_depth,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_depth.append(block_depth)

        blocks_image = torch.nn.Sequential(*blocks_image)
        blocks_depth = torch.nn.Sequential(*blocks_depth)

        return blocks_image, blocks_depth

    def forward(self, image, depth):
        '''
        Forward input x through the ResNet model
        Arg(s):
            image : torch.Tensor
            depth : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = []

        # Resolution 1/1 -> 1/2
        conv1_image = self.conv1_image(image)
        conv1_depth = self.conv1_depth(depth)

        if self.fusion_type == 'add':
            conv1_project = self.conv1_project(conv1_depth)
            conv1 = conv1_project + conv1_image
        elif self.fusion_type == 'weight':
            conv1_weight = self.conv1_weight(conv1_depth)
            conv1 = conv1_weight * conv1_depth + conv1_image
        elif self.fusion_type == 'weight_and_project':
            conv1_weight = self.conv1_weight(conv1_depth)
            conv1_project = self.conv1_project(conv1_depth)
            conv1 = conv1_weight * conv1_project + conv1_image
        elif self.fusion_type == 'concat':
            conv1 = torch.cat([conv1_depth, conv1_image], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(conv1)

        # Resolution 1/2 -> 1/4
        max_pool_image = self.max_pool(conv1_image)
        max_pool_depth = self.max_pool(conv1_depth)

        blocks2_image = self.blocks2_image(max_pool_image)
        blocks2_depth = self.blocks2_depth(max_pool_depth)

        if self.fusion_type == 'add':
            conv2_project = self.conv2_project(blocks2_depth)
            blocks2 = conv2_project + blocks2_image
        elif self.fusion_type == 'weight':
            conv2_weight = self.conv2_weight(blocks2_depth)
            blocks2 = conv2_weight * blocks2_depth + blocks2_image
        elif self.fusion_type == 'weight_and_project':
            conv2_weight = self.conv2_weight(blocks2_depth)
            conv2_project = self.conv2_project(blocks2_depth)
            blocks2 = conv2_weight * conv2_project + blocks2_image
        elif self.fusion_type == 'concat':
            blocks2 = torch.cat([blocks2_image, blocks2_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks2)

        # Resolution 1/4 -> 1/8
        blocks3_image = self.blocks3_image(blocks2_image)
        blocks3_depth = self.blocks3_depth(blocks2_depth)

        if self.fusion_type == 'add':
            conv3_project = self.conv3_project(blocks3_depth)
            blocks3 = conv3_project + blocks3_image
        elif self.fusion_type == 'weight':
            conv3_weight = self.conv3_weight(blocks3_depth)
            blocks3 = conv3_weight * blocks3_depth + blocks3_image
        elif self.fusion_type == 'weight_and_project':
            conv3_weight = self.conv3_weight(blocks3_depth)
            conv3_project = self.conv3_project(blocks3_depth)
            blocks3 = conv3_weight * conv3_project + blocks3_image
        elif self.fusion_type == 'concat':
            blocks3 = torch.cat([blocks3_image, blocks3_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks3)

        # Resolution 1/8 -> 1/16
        blocks4_image = self.blocks4_image(blocks3_image)
        blocks4_depth = self.blocks4_depth(blocks3_depth)

        if self.fusion_type == 'add':
            conv4_project = self.conv4_project(blocks4_depth)
            blocks4 = conv4_project + blocks4_image
        elif self.fusion_type == 'weight':
            conv4_weight = self.conv4_weight(blocks4_depth)
            blocks4 = conv4_weight * blocks4_depth + blocks4_image
        elif self.fusion_type == 'weight_and_project':
            conv4_weight = self.conv4_weight(blocks4_depth)
            conv4_project = self.conv4_project(blocks4_depth)
            blocks4 = conv4_weight * conv4_project + blocks4_image
        elif self.fusion_type == 'concat':
            blocks4 = torch.cat([blocks4_image, blocks4_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks4)

        # Resolution 1/16 -> 1/32
        blocks5_image = self.blocks5_image(blocks4_image)
        blocks5_depth = self.blocks5_depth(blocks4_depth)

        if self.fusion_type == 'add':
            conv5_project = self.conv5_project(blocks5_depth)
            blocks5 = conv5_project + blocks5_image
        elif self.fusion_type == 'weight':
            conv5_weight = self.conv5_weight(blocks5_depth)
            blocks5 = conv5_weight * blocks5_depth + blocks5_image
        elif self.fusion_type == 'weight_and_project':
            conv5_weight = self.conv5_weight(blocks5_depth)
            conv5_project = self.conv5_project(blocks5_depth)
            blocks5 = conv5_weight * conv5_project + blocks5_image
        elif self.fusion_type == 'concat':
            blocks5 = torch.cat([blocks5_image, blocks5_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks5)

        # Resolution 1/32 -> 1/64
        if self.blocks6_image is not None and self.blocks6_depth is not None:
            blocks6_image = self.blocks6_image(blocks5_image)
            blocks6_depth = self.blocks6_depth(blocks5_depth)

            if self.fusion_type == 'add':
                conv6_project = self.conv6_project(blocks6_depth)
                blocks6 = conv6_project + blocks6_image
            elif self.fusion_type == 'weight':
                conv6_weight = self.conv6_weight(blocks6_depth)
                blocks6 = conv6_weight * blocks6_depth + blocks6_image
            elif self.fusion_type == 'weight_and_project':
                conv6_weight = self.conv6_weight(blocks6_depth)
                conv6_project = self.conv6_project(blocks6_depth)
                blocks6 = conv6_weight * conv6_project + blocks6_image
            elif self.fusion_type == 'concat':
                blocks6 = torch.cat([blocks6_image, blocks6_depth], dim=1)
            else:
                raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

            layers.append(blocks6)

        # Resolution 1/64 -> 1/128
        if self.blocks7_image is not None and self.blocks7_depth is not None:
            blocks7_image = self.blocks7_image(blocks6_image)
            blocks7_depth = self.blocks7_depth(blocks6_depth)

            if self.fusion_type == 'add':
                conv7_project = self.conv7_project(blocks7_depth)
                blocks7 = conv7_project + blocks7_image
            elif self.fusion_type == 'weight':
                conv7_weight = self.conv7_weight(blocks7_depth)
                blocks7 = conv7_weight * blocks7_depth + blocks7_image
            elif self.fusion_type == 'weight_and_project':
                conv7_weight = self.conv7_weight(blocks7_depth)
                conv7_project = self.conv7_project(blocks7_depth)
                blocks7 = conv7_weight * conv7_project + blocks7_image
            elif self.fusion_type == 'concat':
                blocks7 = torch.cat([blocks7_image, blocks7_depth], dim=1)
            else:
                raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

            layers.append(blocks7)

        return layers[-1], layers[:-1]

class FullyConnectedEncoder(torch.nn.Module):
    '''
    Fully connected encoder
    Arg(s):
        input_channels : int
            number of input channels
        n_neurons : list[int]
            number of filters to use per layer
        latent_size : int
            number of output neuron
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''

    def __init__(self,
                 input_channels=3,
                 n_neurons=[32, 64, 96, 128, 256],
                 latent_size=29 * 10,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(FullyConnectedEncoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.mlp = torch.nn.Sequential(
            net_utils.FullyConnected(
                in_features=input_channels,
                out_features=n_neurons[0],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[0],
                out_features=n_neurons[1],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[1],
                out_features=n_neurons[2],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[2],
                out_features=n_neurons[3],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[3],
                out_features=n_neurons[4],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[4],
                out_features=latent_size,
                weight_initializer=weight_initializer,
                activation_func=activation_func))

    def forward(self, x):

        return self.mlp(x)


class RadarNet(torch.nn.Module):
    '''
    Radar association network
    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_image : int
            number of filters for image (RGB) branch for each KB layer
        n_filters_depth : int
            number of filters for depth branch  for each KB layer
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=3,
                 n_filters_encoder_image=[32, 64, 128, 128, 128],
                 n_filters_encoder_depth=[32, 64, 128, 128, 128],
                 n_output_depth=29 * 10,
                 n_filters_decoder=[256, 128, 64, 32, 16],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(RadarNet, self).__init__()

        self.latent_channels_depth = n_filters_encoder_depth[-1]

        self.encoder_image = ResNetEncoder(
            n_layer=18,
            input_channels=input_channels_image,
            n_filters=n_filters_encoder_image,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.encoder_depth = FullyConnectedEncoder(
            input_channels=input_channels_depth,
            n_filters=n_filters_encoder_depth,
            n_output=n_output_depth,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        n_skips = n_filters_encoder_image[:-1]
        n_skips = n_skips[::-1] + [0]

        self.decoder = MultiScaleDecoder(
            input_channels=n_filters_encoder_image[-1] + n_filters_encoder_depth[-1],
            output_channels=1,
            n_resolution=1,
            n_filters=n_filters_decoder,
            n_skips=n_skips,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            output_func='linear',
            use_batch_norm=use_batch_norm,
            deconv_type='up')

    def forward(self, image, depth, shape):

        latent_height = torch.ceil(shape[-2] / 32.0).int()
        latent_width = torch.ceil(shape[-1] / 32.0).int()

        latent_image, skips_image = self.encoder_image(image)

        latent_depth = self.encoder_depth(depth)

        latent_depth = latent_depth.view(-1, self.latent_channels_depth, latent_height, latent_width)

        latent = torch.cat([latent_image, latent_depth], dim=1)

        output = self.decoder(latent, skips_image)

        return output[-1]


class RadarNetV1Encoder(torch.nn.Module):
    '''
    Radar association network
    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_encoder_image : int
            number of filters for image (RGB) branch
        n_neurons_encoder_depth : int
            number of neurons for depth (radar) branch
        latent_size_depth : int
            size of latent vector
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=3,
                 input_patch_size_image=(900, 288),
                 n_filters_encoder_image=[32, 64, 128, 128, 128],
                 n_neurons_encoder_depth=[32, 64, 128, 128, 128],
                 latent_size_depth=128 * 29 * 10,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(RadarNetV1Encoder, self).__init__()

        self.n_neuron_latent_depth = n_neurons_encoder_depth[-1]

        self.encoder_image = ResNetEncoder(
            n_layer=18,
            input_channels=input_channels_image,
            n_filters=n_filters_encoder_image,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.encoder_depth = FullyConnectedEncoder(
            input_channels=input_channels_depth,
            n_neurons=n_neurons_encoder_depth,
            latent_size=latent_size_depth,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.input_patch_size_image=input_patch_size_image

    def forward(self, image, points, b_boxes):
        # Image shape: (B, C, H, W) # Should be (B, 3, 768, 288)
        # points shape: (B*K, X)
        # b_boxes: [(K, 4) * B], this should be a list with B elements, and each element is (K, 4) size
        # K is the number of radar points per image
        # X is the radar dimension
        

        # Define dimensions
        shape = self.input_patch_size_image
        latent_height = int(shape[-2] // 32.0)
        latent_width = int(shape[-1] // 32.0)
        batch_size = image.shape[0]

        # Define scales and feature sizes
        skip_scales = [1/2.0, 1/4.0, 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0]
        skip_feature_sizes = [
            (int(shape[-2] * skip_scale), 
             int(shape[-1] * skip_scale)) 
            for skip_scale in skip_scales
        ] # Should be [(384, 144), (192, 72), (96, 36), (48, 18)]
        
        latent_scale = 1/32.0
        latent_feature_size = (latent_height, latent_width) # Should be (24, 9)

        # Forward the entire image
        latent_image, skips_image = self.encoder_image(image)

        # ROI pooling on latent images
        latent_image_pooled = torchvision.ops.roi_pool(
            latent_image, b_boxes, 
            spatial_scale=latent_scale, 
            output_size=latent_feature_size
        ) # (N*K, C, H, W)
        
        # ROI pooling on the skips
        skips_image_pooled = []
        for skip_image_idx in range(len(skips_image)):
            skips_image_pooled.append(
                torchvision.ops.roi_pool(
                    skips_image[skip_image_idx], b_boxes, 
                    spatial_scale=skip_scales[skip_image_idx], 
                    output_size=skip_feature_sizes[skip_image_idx]
                ) # (N*K, C, H, W)
            )
        
        # Radar points
        # points = points.view(-1, points.shape[-1]) # N, K, X -> N*K, X
        latent_depth = self.encoder_depth(points)
        latent_depth = latent_depth.view(points.shape[0], self.n_neuron_latent_depth, -1, latent_width)
        
        # Concatenate the features
        latent = torch.cat([latent_image_pooled, latent_depth], dim=1)
        return latent, skips_image_pooled


class ResNetBasedEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_image : int
            number of filters for image (RGB) branch for each KB layer
        n_filters_depth : int
            number of filters for depth branch  for each KB layer
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels_image=3,
                 input_channels_depth=1,
                 n_filters_image=[48, 96, 192, 384, 384],
                 n_filters_depth=[16, 32, 64, 128, 128],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetBasedEncoder, self).__init__()

        self.encoder_image = ResNetEncoder(
            n_layer=18,
            input_channels=input_channels_image,
            n_filters=n_filters_image,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.encoder_depth = ResNetEncoder(
            n_layer=18,
            input_channels=input_channels_depth,
            n_filters=n_filters_depth,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self,  image, depth):
        '''
        Forward input x through the ResNet model
        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W tensor
            depth : torch.Tensor[float32]
                N x 2 x H x W tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        latent_image, skips_image = self.encoder_image(image)
        latent_depth, skips_depth = self.encoder_depth(depth)

        # Concatenate skips and latents
        latent = torch.cat([latent_image, latent_depth], dim=1)
        skips = [
            torch.cat([skip_image, skip_depth], dim=1)
            for skip_image, skip_depth in zip(skips_image, skips_depth)
        ]

        return latent, skips


'''
Decoder Architectures
'''
class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections
    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : int list
            number of filters to use at each decoder block
        n_skips : int list
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=1,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 deconv_type='up'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 3:
            self.output3 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/8 -> 1/4
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 2:
            self.output2 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/4 -> 1/2
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output0 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network
        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/32 -> 1/16
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1

        layers.append(self.deconv3(layers[-1], skips[n]))

        if self.n_resolution > 3:
            output3 = self.output3(layers[-1])
            outputs.append(output3)

            upsample_output3 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n - 1

        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))

        if self.n_resolution > 2:
            output2 = self.output2(layers[-1])
            outputs.append(output2)

            upsample_output2 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n - 1

        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            upsample_output1 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:

                if n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs