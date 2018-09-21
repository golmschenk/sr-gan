"""
Code for the model structures.
"""
import torch
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, Sequential, BatchNorm2d, Linear, Dropout
from torch.nn.functional import leaky_relu, tanh, max_pool2d

from crowd.data import patch_size
from utility import seed_all


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
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
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
        self.density_layer5 = convolution(conv_dim * 8, int(patch_size / 4) ** 2, int(image_size / 16), 1, 0,
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
        density = self.density_layer5(out).view(-1, int(patch_size / 4), int(patch_size / 4))
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
