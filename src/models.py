import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import net_utils

'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Args:
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

        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        assert(len(n_filters) == len(n_blocks) + 1)

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        in_channels, out_channels = [input_channels, n_filters[0]]
        self.conv1 = net_utils.Conv2d(in_channels, out_channels,
            kernel_size=[17,3],
            stride=[2,2],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        in_channels, out_channels = [n_filters[0], n_filters[1]]

        blocks2 = []
        for n in range(n_blocks[0]):
            if n == 0:
                block = resnet_block(in_channels, out_channels,
                	kernel_size=[17,3],
                    stride=[2,2],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks2.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[1,1],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks2.append(block)
        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        blocks3 = []
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        for n in range(n_blocks[1]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[2,2],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks3.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[1,1],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks3.append(block)
        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        blocks4 = []
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        for n in range(n_blocks[2]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[2,2],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks4.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[1,1],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks4.append(block)
        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        blocks5 = []
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        for n in range(n_blocks[3]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[2,2],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks5.append(block)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels,
                    kernel_size=[17,3],
                    stride=[1,1],
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                blocks5.append(block)
        self.blocks5 = torch.nn.Sequential(*blocks5)

    def forward(self, x):
        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.blocks2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        return layers[-1], layers[1:-1]

class FullConnectedEncoder(nn.Module):
	"""FusionNet"""
	def __init__(self, n_filters=[256, 128, 64, 32], n_outputs=29*10, radar_input_dim=3):
		super(FullConnectedEncoder, self).__init__()
		self.radar_input_dim = radar_input_dim


		self.radar_branch = nn.Sequential(
			nn.Linear(self.radar_input_dim, n_filters[0]),
			nn.BatchNorm1d(n_filters[0]),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Linear(n_filters[0], n_filters[1]),
			nn.BatchNorm1d(n_filters[1]),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Linear(n_filters[1],n_filters[2]),
			nn.BatchNorm1d(n_filters[2]),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Linear(n_filters[2],n_filters[3]*n_outputs),
			nn.BatchNorm1d(n_filters[3]*n_outputs),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
		)

	def forward(self, radar_input):
		radar_features = self.radar_branch(radar_input) # B, 256*29*2
		return radar_features


class VOICEDDecoder(torch.nn.Module):
    '''
   VOICED decoder with skip connections

    Args:
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
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
                 n_filters=[256, 128, 64, 32],
                 n_skips=[256, 128, 64, 32],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 deconv_type='transpose'):
        super(VOICEDDecoder, self).__init__()

        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[0], n_filters[0]
        ]
        self.deconv4 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels,
        	kernel_size=[3,3],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [
            n_filters[0], n_skips[1], n_filters[1]
        ]
        self.deconv3 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels,
        	kernel_size=[3,3],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [
            n_filters[1], n_skips[2], n_filters[2]
        ]
        self.deconv2 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels,
        	kernel_size=[3,3],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [
            n_filters[2], n_skips[3], n_filters[3]
        ]
        if deconv_type == 'transpose':
            self.deconv1 = net_utils.TransposeConv2d(in_channels, out_channels,
                kernel_size=[3,3],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif deconv_type == 'up':
            self.deconv1 = net_utils.UpConv2d(in_channels, out_channels,
                kernel_size=[3,3],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

        concat_channels = skip_channels + out_channels
        self.output1 = net_utils.Conv2d(concat_channels, output_channels,
            kernel_size=[3,3],
            stride=[1,1],
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

    def forward(self, x, skips):
        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips) - 1
        layers.append(self.deconv4(layers[-1], skips[n]))
        

        # Resolution 1/16 -> 1/8
        n = n - 1
        layers.append(self.deconv3(layers[-1], skips[n]))

        # Resolution 1/8 -> 1/4
        n = n - 1
        layers.append(self.deconv2(layers[-1], skips[n]))

        # Resolution 1/4 -> 1/2
        n = n - 1
        layers.append(self.deconv1(layers[-1], skips[n].shape[-2:]))
        concat = torch.cat([layers[-1], skips[n]], dim=1)

        output1 = self.output1(concat)
        upsample_output1 = torch.nn.functional.interpolate(output1,
            scale_factor=2, mode='nearest')

        # Resolution 1/2 -> 1/1
        outputs.append(upsample_output1)

        return outputs


class FusionNet(nn.Module):
    """FusionNet"""
    def __init__(self, bias_factor=0, radar_input_dim=3):
        super(FusionNet, self).__init__()
        self.radar_input_dim = radar_input_dim
        self.bias_factor = bias_factor
        self.image_encoder = ResNetEncoder(n_layer=18,
			input_channels=3,
	        n_filters=[32, 64, 128, 128, 128],
            weight_initializer='kaiming_uniform',
            activation_func='leaky_relu',
            use_batch_norm=True,)
        self.radar_branch = FullConnectedEncoder(n_filters=[32, 64, 128, 128], n_outputs=29*10)
        self.decoder = VOICEDDecoder(input_channels=256,
            output_channels=1,
            n_filters=[256, 128, 64, 32],
            n_skips=[128, 128, 64, 32],
            weight_initializer='kaiming_uniform',
            activation_func='leaky_relu',
            output_func='linear',
            use_batch_norm=True,
            deconv_type='up')
        
    def forward(self, img_input, radar_input):
        img_features, skips = self.image_encoder(img_input)
        radar_features = self.radar_branch(radar_input) # B, 128*29*2
        radar_features = radar_features.view(-1,128,29,10)
        merged_features = torch.cat((img_features, radar_features), dim=1) # B, 128
        decoded_features = self.decoder(merged_features, skips)
        bias = torch.ones_like(decoded_features[-1]) # we are adding a bias to force our model to learn larger values for the corresponding surfaces
        est_logits = decoded_features[-1] - self.bias_factor*bias

        return est_logits

