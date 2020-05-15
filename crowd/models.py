"""
Code for the model structures.
"""
import re
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, Sequential, BatchNorm2d, Linear, Dropout, \
    BatchNorm2d
from torch.nn.functional import leaky_relu, max_pool2d, dropout, avg_pool2d, relu
from torch import tanh
from torch.utils import model_zoo
import torchvision.models.densenet

from utility import seed_all, gpu


class JointCNN(Module):
    """
    A CNN that produces a density map and a count.
    """
    def __init__(self):
        seed_all(0)
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=7, padding=3)
        self.max_pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7, padding=3)
        self.max_pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5, padding=2)
        self.conv4 = Conv2d(self.conv3.out_channels, 1000, kernel_size=18)
        self.conv5 = Conv2d(self.conv4.out_channels, 400, kernel_size=1)
        self.count_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)
        self.density_conv = Conv2d(self.conv5.out_channels, 324, kernel_size=1)
        self.features = None

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        """
        The forward pass of the network.

        :param x: The input images.
        :type x: torch.autograd.Variable
        :return: The predicted density labels.
        :rtype: torch.autograd.Variable
        """
        patch_size = x.shape[2]
        x = leaky_relu(self.conv1(x))
        x = self.max_pool1(x)
        x = leaky_relu(self.conv2(x))
        x = self.max_pool2(x)
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = leaky_relu(self.conv5(x))
        self.features = x
        x_count = leaky_relu(self.count_conv(x)).view(-1)
        x_density = leaky_relu(self.density_conv(x)).view(-1, int(patch_size / 4), int(patch_size / 4))
        return x_density, x_count


class Generator(Module):
    """
    A generator for producing crowd images.
    """
    def __init__(self):
        seed_all(0)
        super().__init__()
        self.input_size = 100
        self.conv_transpose1 = ConvTranspose2d(self.input_size, 64, kernel_size=18)
        self.conv_transpose2 = ConvTranspose2d(self.conv_transpose1.out_channels, 32, kernel_size=4, stride=2,
                                               padding=1)
        self.conv_transpose3 = ConvTranspose2d(self.conv_transpose2.out_channels, 3, kernel_size=4, stride=2,
                                               padding=1)

    def forward(self, z):
        """
        The forward pass of the generator.

        :param z: The input images.
        :type z: torch.autograd.Variable
        :return: Generated images.
        :rtype: torch.autograd.Variable
        """
        z = z.view(-1, self.input_size, 1, 1)
        z = leaky_relu(self.conv_transpose1(z))
        z = leaky_relu(self.conv_transpose2(z))
        z = tanh(self.conv_transpose3(z))
        return z

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)


batch_norm = False


def transpose_convolution(c_in, c_out, k_size, stride=2, pad=1, bn=batch_norm):
    """A transposed convolution layer."""
    layers = [ConvTranspose2d(c_in, c_out, k_size, stride, pad)]
    if bn:
        layers.append(BatchNorm2d(c_out))
    return Sequential(*layers)


def convolution(c_in, c_out, k_size, stride=2, pad=1, bn=batch_norm):
    """A convolutional layer."""
    layers = [Conv2d(c_in, c_out, k_size, stride, pad)]
    if bn:
        layers.append(BatchNorm2d(c_out))
    return Sequential(*layers)


class DCGenerator(Module):
    """A DCGAN-like generator architecture."""
    def __init__(self, z_dim=256, image_size=224, conv_dim=64):
        seed_all(0)
        super().__init__()
        self.fc = transpose_convolution(z_dim, conv_dim * 8, int(image_size / 16), 1, 0, bn=False)
        self.layer1 = transpose_convolution(conv_dim * 8, conv_dim * 4, 4)
        self.layer2 = transpose_convolution(conv_dim * 4, conv_dim * 2, 4)
        self.layer3 = transpose_convolution(conv_dim * 2, conv_dim, 4)
        self.layer4 = transpose_convolution(conv_dim, 3, 4, bn=False)
        self.input_size = z_dim

    def forward(self, z):
        """The forward pass of the network."""
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = leaky_relu(self.layer1(out), 0.05)    # (?, 256, 8, 8)
        out = leaky_relu(self.layer2(out), 0.05)    # (?, 128, 16, 16)
        out = leaky_relu(self.layer3(out), 0.05)    # (?, 64, 32, 32)
        out = tanh(self.layer4(out))                # (?, 3, 64, 64)
        return out


class JointDCDiscriminator(Module):
    """A DCGAN-like discriminator architecture."""
    def __init__(self, image_size=128, conv_dim=64, number_of_outputs=1):
        seed_all(0)
        super().__init__()
        self.number_of_outputs = number_of_outputs
        self.layer1 = convolution(3, conv_dim, 4, bn=False)
        self.layer2 = convolution(conv_dim, conv_dim * 2, 4)
        self.layer3 = convolution(conv_dim * 2, conv_dim * 4, 4)
        self.layer4 = convolution(conv_dim * 4, conv_dim * 8, 4)
        self.count_layer5 = convolution(conv_dim * 8, self.number_of_outputs, int(image_size / 16), 1, 0, False)
        self.density_layer5 = convolution(conv_dim * 8, int(image_size / 4) ** 2, int(image_size / 16), 1, 0,
                                          False)
        self.features = None

    def forward(self, x):
        """The forward pass of the network."""
        out = leaky_relu(self.layer1(x), 0.05)    # (?, 64, 32, 32)
        out = leaky_relu(self.layer2(out), 0.05)  # (?, 128, 16, 16)
        out = leaky_relu(self.layer3(out), 0.05)  # (?, 256, 8, 8)
        out = leaky_relu(self.layer4(out), 0.05)  # (?, 512, 4, 4)
        self.features = out.view(out.size(0), -1)
        count = self.count_layer5(out).view(-1)
        if self.number_of_outputs == 1:
            count = count.view(-1)
        else:
            count = count.view(-1, self.number_of_outputs)
        density = self.density_layer5(out).view(-1, int(x.shape[2] / 4), int(x.shape[2] / 4))
        return density, count


def spatial_pyramid_pooling(input_, output_size):
    """
    Adds a spatial pyramid pooling layer.

    :param input_: The input to the layer.
    :type input_: torch.Tensor
    :param output_size: The output size of the layer (number of pooling grid cells).
    :type output_size: int
    :return: The pooling layer.
    :rtype: torch.nn.Module
    """
    assert input_.dim() == 4 and input_.size(2) == input_.size(3)
    kernel_size = input_.size(2) // output_size
    padding = 0
    if input_.size(2) // kernel_size > output_size:
        kernel_size += 1
        padding = 1
    return max_pool2d(input_, kernel_size=kernel_size, padding=padding)


class SpatialPyramidPoolingDiscriminator(Module):
    """A discriminator that uses spatial pyramid pooling as a primary feature."""
    def __init__(self, image_size=128):
        seed_all(0)
        super().__init__()
        self.density_label_size = image_size // 4
        self.conv1 = Conv2d(3, 16, kernel_size=7)
        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7)
        self.max_pool2 = MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5)
        self.conv4 = Conv2d(self.conv3.out_channels, 32, kernel_size=3)
        self.conv5 = Conv2d(self.conv4.out_channels, 16, kernel_size=3)
        self.conv6 = Conv2d(self.conv5.out_channels, 16, kernel_size=3, dilation=2)
        self.conv7 = Conv2d(self.conv6.out_channels, 16, kernel_size=3, dilation=2)
        # Feature 5 regression
        self.f5_fc1 = Linear(912, 1000)
        self.f5_density = Linear(1000, self.density_label_size ** 2)
        self.f5_count = Linear(1000, 1)
        # Feature 7 regression
        self.f7_fc1 = Linear(912, 1000)
        self.f7_density = Linear(1000, self.density_label_size ** 2)
        self.f7_count = Linear(1000, 1)
        self.features = None

    def forward(self, x):
        """The forward pass of the network."""
        out = leaky_relu(self.conv1(x))
        out = self.max_pool1(out)
        out = leaky_relu(self.conv2(out))
        out = self.max_pool2(out)
        out = leaky_relu(self.conv3(out))
        out = leaky_relu(self.conv4(out))
        out5 = leaky_relu(self.conv5(out))
        out = leaky_relu(self.conv6(out5))
        out7 = leaky_relu(self.conv7(out))

        f5_1 = spatial_pyramid_pooling(out5, 1).view(-1, 1 * 16)
        f5_2 = spatial_pyramid_pooling(out5, 2).view(-1, 4 * 16)
        f5_4 = spatial_pyramid_pooling(out5, 4).view(-1, 16 * 16)
        f5_6 = spatial_pyramid_pooling(out5, 6).view(-1, 36 * 16)
        f5 = torch.cat([f5_1, f5_2, f5_4, f5_6], dim=1)
        f5 = leaky_relu(self.f5_fc1(f5))
        f5_density = leaky_relu(self.f5_density(f5))
        f5_count = leaky_relu(self.f5_count(f5))

        f7_1 = spatial_pyramid_pooling(out7, 1).view(-1, 1 * 16)
        f7_2 = spatial_pyramid_pooling(out7, 2).view(-1, 4 * 16)
        f7_4 = spatial_pyramid_pooling(out7, 4).view(-1, 16 * 16)
        f7_6 = spatial_pyramid_pooling(out7, 6).view(-1, 36 * 16)
        f7 = torch.cat([f7_1, f7_2, f7_4, f7_6], dim=1)
        f7 = leaky_relu(self.f7_fc1(f7))
        f7_density = leaky_relu(self.f7_density(f7))
        f7_count = leaky_relu(self.f7_count(f7))

        self.features = torch.cat([f5, f7], dim=1)
        density = f5_density + f7_density
        density = density.view(-1, self.density_label_size, self.density_label_size)
        count = f5_count + f7_count
        count = count.view(-1)
        return density, count


class FullSpatialPyramidPoolingDiscriminator(Module):
    """A discriminator that uses spatial pyramid pooling as a primary feature."""
    def __init__(self, image_size=128):
        seed_all(0)
        super().__init__()
        self.density_label_size = image_size // 4
        self.conv1 = Conv2d(3, 16, kernel_size=7)
        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7)
        self.bn2 = BatchNorm2d(self.conv2.out_channels)
        self.max_pool2 = MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5)
        self.bn3 = BatchNorm2d(self.conv3.out_channels)
        self.conv4 = Conv2d(self.conv3.out_channels, 32, kernel_size=3)
        self.bn4 = BatchNorm2d(self.conv4.out_channels)
        self.conv5 = Conv2d(self.conv4.out_channels, 16, kernel_size=3)
        self.bn5 = BatchNorm2d(self.conv5.out_channels)
        self.conv6 = Conv2d(self.conv5.out_channels, 16, kernel_size=3, dilation=2)
        self.bn6 = BatchNorm2d(self.conv6.out_channels)
        self.conv7 = Conv2d(self.conv6.out_channels, 16, kernel_size=3, dilation=2)
        self.bn7 = BatchNorm2d(self.conv7.out_channels)
        # Feature 5 regression
        self.f5_fc1 = Linear(912, 1000)
        self.f5_dropout = Dropout()
        self.f5_density = Linear(1000, self.density_label_size ** 2)
        self.f5_count = Linear(1000, 1)
        # Feature 7 regression
        self.f7_fc1 = Linear(912, 1000)
        self.f7_dropout = Dropout()
        self.f7_density = Linear(1000, self.density_label_size ** 2)
        self.f7_count = Linear(1000, 1)
        self.features = None

    def forward(self, x):
        """The forward pass of the network."""
        out = leaky_relu(self.conv1(x))
        out = self.max_pool1(out)
        out = leaky_relu(self.bn2(self.conv2(out)))
        out = self.max_pool2(out)
        out = leaky_relu(self.bn3(self.conv3(out)))
        out = leaky_relu(self.bn4(self.conv4(out)))
        out5 = leaky_relu(self.bn5(self.conv5(out)))
        out = leaky_relu(self.bn6(self.conv6(out5)))
        out7 = leaky_relu(self.bn7(self.conv7(out)))

        f5_1 = spatial_pyramid_pooling(out5, 1).view(-1, 1 * 16)
        f5_2 = spatial_pyramid_pooling(out5, 2).view(-1, 4 * 16)
        f5_4 = spatial_pyramid_pooling(out5, 4).view(-1, 16 * 16)
        f5_6 = spatial_pyramid_pooling(out5, 6).view(-1, 36 * 16)
        f5 = torch.cat([f5_1, f5_2, f5_4, f5_6], dim=1)
        f5_do = self.f5_dropout(leaky_relu(self.f5_fc1(f5)))
        f5_density = leaky_relu(self.f5_density(f5_do))
        f5_count = leaky_relu(self.f5_count(f5_do))

        f7_1 = spatial_pyramid_pooling(out7, 1).view(-1, 1 * 16)
        f7_2 = spatial_pyramid_pooling(out7, 2).view(-1, 4 * 16)
        f7_4 = spatial_pyramid_pooling(out7, 4).view(-1, 16 * 16)
        f7_6 = spatial_pyramid_pooling(out7, 6).view(-1, 36 * 16)
        f7 = torch.cat([f7_1, f7_2, f7_4, f7_6], dim=1)
        f7_do = self.f7_dropout(leaky_relu(self.f7_fc1(f7)))
        f7_density = leaky_relu(self.f7_density(f7_do))
        f7_count = leaky_relu(self.f7_count(f7_do))

        self.features = torch.cat([f5, f7], dim=1)
        density = f5_density + f7_density
        density = density.view(-1, self.density_label_size, self.density_label_size)
        count = f5_count + f7_count
        count = count.view(-1)
        return density, count


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        """Forward pass."""
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class RegressionModule(nn.Module):
    """A regression module to output both density map and count value with fully connected layers."""
    def __init__(self, in_features, label_patch_size):
        super().__init__()
        self.fc = Linear(in_features, 1000)
        self.fc_density = Linear(1000, label_patch_size ** 2)
        self.fc_count = Linear(1000, 1)

    def forward(self, x):
        """Forward pass."""
        fc_out = leaky_relu(self.fc(x))
        fc_density_out = leaky_relu(self.fc_density(fc_out))
        fc_count = leaky_relu(self.fc_count(fc_out))
        return fc_density_out, fc_count


class SppModule(nn.Module):
    """A basic spatial pyramid pooling module."""
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        """Forward pass."""
        p1 = spatial_pyramid_pooling(x, 1).view(-1, 1 * self.in_features)
        p2 = spatial_pyramid_pooling(x, 2).view(-1, 4 * self.in_features)
        p3 = spatial_pyramid_pooling(x, 4).view(-1, 16 * self.in_features)
        p4 = spatial_pyramid_pooling(x, 6).view(-1, 36 * self.in_features)
        out = torch.cat([p1, p2, p3, p4], dim=1)
        return out


class SppDenseNet(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=28):

        super(SppDenseNet, self).__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.spp1 = SppModule(128)
        self.regression_module1 = RegressionModule(7296, label_patch_size=self.label_patch_size)
        self.final_regression_module = RegressionModule(1920, label_patch_size=self.label_patch_size)

    def forward(self, x):
        """Forward pass."""
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1).view(n5_out.size(0), -1)

        spp1_out = self.spp1(t1_out)
        rm1_density, rm1_count = self.regression_module1(spp1_out)
        final_density, final_count = self.final_regression_module(final_pool)
        count = rm1_count + final_count
        density = rm1_density + final_density
        count = count.view(-1)
        density = density.view(-1, self.label_patch_size, self.label_patch_size)
        return density, count


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.features = None

        # First convolution
        self.dense_layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_layers.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.dense_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.dense_layers.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        """Forward pass."""
        features = self.dense_layers(x)
        out = relu(features, inplace=True)
        out = avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        self.features = out
        out = self.classifier(out)
        return out


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    This copied directly from the model zoo removing the count_layer to fix the class_num change bug
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        new_name_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace('features.denseblock', 'dense_layers.denseblock')
            new_name_state_dict[new_key] = value
        state_dict = new_name_state_dict
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    return model


class DenseNetDiscriminator(Module):
    """The DenseNet as a discriminator."""
    def __init__(self, label_patch_size=224):
        seed_all(0)
        super().__init__()
        self.label_patch_size = label_patch_size
        self.features = None
        self.dense_net_module = densenet201(pretrained=True, num_classes=1)

    def forward(self, x):
        """The forward pass of the network."""
        batch_size = x.shape[0]
        out = self.dense_net_module(x)
        self.features = self.dense_net_module.features
        out = out.view(-1)
        return (torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu), out,
                torch.zeros([batch_size, 3, self.label_patch_size, self.label_patch_size], device=gpu))


class DenseNetDiscriminatorDggan(Module):
    """The DenseNet as a discriminator."""

    def __init__(self, label_patch_size=224):
        seed_all(0)
        super().__init__()
        self.label_patch_size = label_patch_size
        self.features = None
        self.dense_net_module = densenet201(pretrained=True, num_classes=2)

    def forward(self, x):
        """The forward pass of the network."""
        batch_size = x.shape[0]
        out = self.dense_net_module(x)
        self.features = self.dense_net_module.features
        out = out.view(-1, 2)
        count, real_label = out[:, 0].squeeze(), out[:, 1].squeeze()
        self.real_label = real_label
        return (torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu), count,
                torch.zeros([batch_size, 3, self.label_patch_size, self.label_patch_size], device=gpu))


class KnnDenseNet(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=28):

        super(KnnDenseNet, self).__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.count_layer = Conv2d(in_channels=num_features, out_channels=1, kernel_size=1)
        self.knn1_layer = Conv2d(in_channels=128, out_channels=1, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1)

        density = torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu)
        count = leaky_relu(self.count_layer(final_pool)).view(batch_size)
        map_ = leaky_relu(self.knn1_layer(t1_out)).view(batch_size, self.label_patch_size, self.label_patch_size)
        return density, count, map_


class MapModule(nn.Module):
    """A module to upscale to a map and produce a count."""
    def __init__(self, in_features, input_size, label_size):
        super().__init__()
        kernel_size = label_size // input_size
        self.map_transposed_conv_layer = ConvTranspose2d(in_channels=in_features, out_channels=1,
                                                         kernel_size=kernel_size, stride=kernel_size)
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        self.conv3 = Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        count_layer_kernel_size = label_size // (2 ** 3)
        self.count_layer = Conv2d(in_channels=32, out_channels=1, kernel_size=count_layer_kernel_size)

    def forward(self, x):
        """Forward pass."""
        map_ = leaky_relu(self.map_transposed_conv_layer(x))
        out = leaky_relu(self.conv1(map_))
        out = leaky_relu(self.conv2(out))
        out = leaky_relu(self.conv3(out))
        count = leaky_relu(self.count_layer(out))
        return map_, count, out


class KnnDenseNet2(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=28):

        super().__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.count_layer = Conv2d(in_channels=num_features, out_channels=1, kernel_size=1)
        self.map_module1 = MapModule(in_features=128, input_size=28, label_size=label_patch_size)
        self.map_module2 = MapModule(in_features=256, input_size=14, label_size=label_patch_size)
        self.map_module3 = MapModule(in_features=896, input_size=7, label_size=label_patch_size)

    def forward(self, x):
        """Forward pass."""
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1)

        density = torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu)
        final_count = leaky_relu(self.count_layer(final_pool))
        map1, count1 = self.module1(t1_out)
        map2, count2 = self.map_module2(t2_out)
        map3, count3 = self.map_module3(t3_out)
        count = count1 + count2 + count3 + final_count
        count = count.view(batch_size)
        map_ = map1 + map2 + map3
        map_ = map_.view(batch_size, self.label_patch_size, self.label_patch_size)
        return density, count, map_


class MapModuleDggan(nn.Module):
    """A module to upscale to a map and produce a count."""
    def __init__(self, in_features, input_size, label_size):
        super().__init__()
        kernel_size = label_size // input_size
        self.map_transposed_conv_layer = ConvTranspose2d(in_channels=in_features, out_channels=1,
                                                         kernel_size=kernel_size, stride=kernel_size)
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        self.conv3 = Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        count_layer_kernel_size = label_size // (2 ** 3)
        self.linear1 = Conv2d(in_channels=32, out_channels=20, kernel_size=count_layer_kernel_size)
        self.count_layer = Conv2d(in_channels=20, out_channels=2, kernel_size=1)


    def forward(self, x):
        """Forward pass."""
        map_ = leaky_relu(self.map_transposed_conv_layer(x))
        out = leaky_relu(self.conv1(map_))
        out = leaky_relu(self.conv2(out))
        out = leaky_relu(self.conv3(out))
        out = leaky_relu(self.linear1(out))
        count = self.count_layer(out)
        return map_, count, out


class KnnDenseNetCatDggan(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=224):

        super().__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.map_module1 = MapModuleDggan(in_features=128, input_size=28, label_size=label_patch_size)
        self.map_module2 = MapModuleDggan(in_features=256, input_size=14, label_size=label_patch_size)
        self.map_module3 = MapModuleDggan(in_features=896, input_size=7, label_size=label_patch_size)
        self.final_count_feature_layer = Conv2d(in_channels=num_features, out_channels=20, kernel_size=1)
        self.count_layer = Conv2d(in_channels=20, out_channels=2, kernel_size=1)
        self.features = None
        self.real_label = None

    def forward(self, x):
        """Forward pass."""
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1)

        density = torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu)
        final_count_features = leaky_relu(self.final_count_feature_layer(final_pool))
        final_count = self.count_layer(final_count_features)
        map1, count1, h1 = self.map_module1(t1_out)
        map2, count2, h2 = self.map_module2(t2_out)
        map3, count3, h3 = self.map_module3(t3_out)
        count = count1 + count2 + count3 + final_count
        count = count.view(batch_size, 2)
        count, real_label = count[:, 0].squeeze(), count[:, 1].squeeze()
        self.real_label = real_label
        map_ = torch.cat([map1, map2, map3], dim=1)
        map_ = map_.view(batch_size, 3, self.label_patch_size, self.label_patch_size)
        return density, count, map_


class KnnDenseNetCat(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=224):

        super().__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.map_module1 = MapModule(in_features=128, input_size=28, label_size=label_patch_size)
        self.map_module2 = MapModule(in_features=256, input_size=14, label_size=label_patch_size)
        self.map_module3 = MapModule(in_features=896, input_size=7, label_size=label_patch_size)
        self.count_layer = Conv2d(in_channels=num_features, out_channels=1, kernel_size=1)
        self.features = None

    def forward(self, x):
        """Forward pass."""
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1)

        density = torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu)
        final_count = leaky_relu(self.count_layer(final_pool))
        map1, count1, h1 = self.map_module1(t1_out)
        map2, count2, h2 = self.map_module2(t2_out)
        map3, count3, h3 = self.map_module3(t3_out)
        count = count1 + count2 + count3 + final_count
        count = count.view(batch_size)
        map_ = torch.cat([map1, map2, map3], dim=1)
        map_ = map_.view(batch_size, 3, self.label_patch_size, self.label_patch_size)
        return density, count, map_


class DenseMapModule(nn.Module):
    """A dense block followed by a map module."""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, label_patch_size):
        super().__init__()
        self.dense_block = _DenseBlock(num_layers=num_layers, num_input_features=num_input_features, bn_size=bn_size,
                                       growth_rate=growth_rate, drop_rate=drop_rate)
        self.map_module = MapModule(in_features=num_input_features + 128, input_size=28, label_size=label_patch_size)

    def forward(self, x):
        """Forward pass."""
        dense_block_output = self.dense_block(x)
        map_, count = self.map_module(leaky_relu(dense_block_output))
        return map_, count, dense_block_output


class KnnDenseNetCatBranch(nn.Module):
    r"""A spatial pooling pyramid network based on DenseNet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 48, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True,
                 label_patch_size=28):

        super().__init__()
        self.label_patch_size = label_patch_size

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        # First convolution
        self.conv_layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_blocks.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if pretrained:
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(torchvision.models.densenet.model_urls['densenet201'])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            new_name_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('features.denseblock', 'dense_blocks.denseblock')
                new_key = new_key.replace('features.transition', 'transition_layers.transition')
                if 'norm5' in new_key:
                    new_key = new_key.replace('features.', '')
                else:
                    new_key = new_key.replace('features.', 'conv_layer1.')
                new_name_state_dict[new_key] = value
            state_dict = new_name_state_dict
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            self.load_state_dict(state_dict, strict=True)

        self.count_layer = Conv2d(in_channels=num_features, out_channels=1, kernel_size=1)
        self.map_module1 = MapModule(in_features=512, input_size=28, label_size=label_patch_size)
        self.density_module1 = DenseMapModule(num_layers=4, num_input_features=513, bn_size=bn_size,
                                              growth_rate=growth_rate, drop_rate=drop_rate,
                                              label_patch_size=label_patch_size)
        self.density_module2 = DenseMapModule(num_layers=4, num_input_features=642, bn_size=bn_size,
                                              growth_rate=growth_rate, drop_rate=drop_rate,
                                              label_patch_size=label_patch_size)

    def forward(self, x):
        """Forward pass."""
        batch_size = x.shape[0]
        out = self.conv_layer1(x)
        db1_out = self.dense_blocks.denseblock1(out)
        t1_out = self.transition_layers.transition1(db1_out)
        db2_out = self.dense_blocks.denseblock2(t1_out)
        t2_out = self.transition_layers.transition2(db2_out)
        db3_out = self.dense_blocks.denseblock3(t2_out)
        t3_out = self.transition_layers.transition3(db3_out)
        db4_out = self.dense_blocks.denseblock4(t3_out)
        n5_out = self.norm5(db4_out)
        n5_relu_out = relu(n5_out, inplace=True)
        final_pool = avg_pool2d(n5_relu_out, kernel_size=7, stride=1)

        density = torch.zeros([batch_size, self.label_patch_size, self.label_patch_size], device=gpu)
        final_count = leaky_relu(self.count_layer(final_pool))

        map1, count1 = self.map_module1(leaky_relu(db2_out))
        density_module_in1 = torch.cat([map1, db2_out], dim=1)
        map2, count2, density_module_out1 = self.density_module1(density_module_in1)
        density_module_in2 = torch.cat([map2, density_module_out1], dim=1)
        map3, count3, _ = self.density_module2(density_module_in2)

        count = count1 + count2 + count3 + final_count
        count = count.view(batch_size)
        map_ = torch.cat([map1, map2, map3], dim=1)
        map_ = map_.view(batch_size, 3, self.label_patch_size, self.label_patch_size)
        return density, count, map_
