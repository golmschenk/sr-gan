import torch
from torch.autograd import Variable
from torch.nn import Module, Linear, Conv2d, Sequential, MaxPool2d, LeakyReLU
from torch.nn.functional import leaky_relu
from torch.utils import model_zoo

from utility import gpu, seed_all


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(Module):
    """
    Slightly modified version from here: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            LeakyReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = Sequential(
            Linear(256 * 6 * 6, 4096),
            LeakyReLU(inplace=True),
            Linear(4096, 4096),
            LeakyReLU(inplace=True),
            Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


class Generator(Module):
    """The generator model."""
    def __init__(self):
        super().__init__()
        self.input_size = 10
        self.linear1 = Linear(self.input_size, 20)
        self.linear5 = Linear(20, 30)
        self.linear6 = Linear(30, observation_count * irrelevant_data_multiplier)

    def forward(self, x, add_noise=False):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear5(x))
        x = self.linear6(x)
        return x