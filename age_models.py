"""
DCGAN code taken from:
https://github.com/sdhnshu/pytorch-model-zoo/blob/master/dcgan/model.py
Because they had a fairly well written, simple version available.
"""
import torch

import torch.nn as nn
import torch.nn.functional as F

from utility import gpu, seed_all


batch_norm = False


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=batch_norm):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=batch_norm):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
        seed_all(0)
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim * 8,
                         int(image_size / 16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        self.input_size = z_dim

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(out))             # (?, 3, 64, 64)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        seed_all(0)
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, 1, int(image_size / 16), 1, 0, False)
        self.feature_layer = None

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        self.feature_layer = out.view(out.size(0), -1)
        out = self.conv5(out).squeeze()
        return out
