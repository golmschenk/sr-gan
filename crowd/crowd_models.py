"""
Code for the model structures.
"""
import torch
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, Sequential, BatchNorm2d
from torch.nn.functional import leaky_relu, tanh

from crowd.crowd_data import resized_patch_size
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
        self.feature_layer = None

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
        self.feature_layer = x
        x_count = leaky_relu(self.count_conv(x)).view(-1)
        x_density = leaky_relu(self.density_conv(x)).view(-1, int(resized_patch_size / 4), int(resized_patch_size / 4))
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
    layers = [ConvTranspose2d(c_in, c_out, k_size, stride, pad)]
    if bn:
        layers.append(BatchNorm2d(c_out))
    return Sequential(*layers)


def convolution(c_in, c_out, k_size, stride=2, pad=1, bn=batch_norm):
    layers = [Conv2d(c_in, c_out, k_size, stride, pad)]
    if bn:
        layers.append(BatchNorm2d(c_out))
    return Sequential(*layers)


class DCGenerator(Module):
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
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = leaky_relu(self.layer1(out), 0.05)    # (?, 256, 8, 8)
        out = leaky_relu(self.layer2(out), 0.05)    # (?, 128, 16, 16)
        out = leaky_relu(self.layer3(out), 0.05)    # (?, 64, 32, 32)
        out = tanh(self.layer4(out))                # (?, 3, 64, 64)
        return out


class JointDCDiscriminator(Module):
    def __init__(self, image_size=128, conv_dim=64):
        seed_all(0)
        super().__init__()
        self.layer1 = convolution(3, conv_dim, 4, bn=False)
        self.layer2 = convolution(conv_dim, conv_dim * 2, 4)
        self.layer3 = convolution(conv_dim * 2, conv_dim * 4, 4)
        self.layer4 = convolution(conv_dim * 4, conv_dim * 8, 4)
        self.count_layer5 = convolution(conv_dim * 8, 1, int(image_size / 16), 1, 0, False)
        self.density_layer5 = convolution(conv_dim * 8, int(resized_patch_size / 4) ** 2, int(image_size / 16), 1, 0, False)
        self.feature_layer = None

    def forward(self, x):
        out = leaky_relu(self.layer1(x), 0.05)    # (?, 64, 32, 32)
        out = leaky_relu(self.layer2(out), 0.05)  # (?, 128, 16, 16)
        out = leaky_relu(self.layer3(out), 0.05)  # (?, 256, 8, 8)
        out = leaky_relu(self.layer4(out), 0.05)  # (?, 512, 4, 4)
        self.feature_layer = out.view(out.size(0), -1)
        count = self.count_layer5(out).view(-1)
        density = self.density_layer5(out).view(-1, int(resized_patch_size / 4), int(resized_patch_size / 4))
        return density, count
