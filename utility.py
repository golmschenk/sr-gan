import random
import re
import time

import numpy as np
import torch
from scipy.stats import rv_continuous
from tensorboardX import SummaryWriter as SummaryWriter_

class SummaryWriter(SummaryWriter_):
    """A custom version of the Tensorboard summary writer class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.summary_period = 1

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add a scalar to the Tensorboard summary."""
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_scalar(tag, scalar_value, global_step)

    def add_histogram(self, tag, values, global_step=None, bins='auto'):
        """Add a histogram to the Tensorboard summary."""
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_histogram(tag, values, global_step, bins)

    def add_image(self, tag, img_tensor, global_step=None):
        """Add an image to the Tensorboard summary."""
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_image(tag, img_tensor, global_step)

    def is_summary_step(self):
        return self.step % self.summary_period == 0


def infinite_iter(dataset):
    """Create an infinite generator from a dataset."""
    while True:
        for examples in dataset:
            yield examples


def clean_scientific_notation(string):
    """Cleans up scientific notation to remove unneeded fluff digits."""
    regex = r'\.?0*e([+\-])0*([0-9])'
    string = re.sub(regex, r'e\g<1>\g<2>', string)
    string = re.sub(r'e\+', r'e', string)
    return string


def shuffled(list_):
    """A simple wrapper to make a *returning* version of random.shuffle()"""
    random.seed()
    random.shuffle(list_)
    return list_


def gpu(element):
    """
    Moves the element to the GPU if available.

    :param element: The element to move to the GPU.
    :type element: torch.Tensor | torch.nn.Module
    :return: The element moved to the GPU.
    :rtype: torch.Tensor | torch.nn.Module
    """
    if torch.cuda.is_available():
        return element.cuda()
    else:
        return element


def cpu(element):
    """
    Moves the element to the CPU if GPU is available.

    :param element: The element to move to the CPU.
    :type element: torch.Tensor | torch.nn.Module
    :return: The element moved to the CPU.
    :rtype: torch.Tensor | torch.nn.Module
    """
    if torch.cuda.is_available():
        return element.cpu()
    else:
        return element


def load(model_path):
    """
    Loads a model, and if GPU is not available, insures that the model only loads onto CPU.

    :param model_path: The path to the model to be loaded.
    :type model_path: str
    :return: The loaded model.
    :rtype: dict[T]
    """
    if torch.cuda.is_available():
        return torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        return torch.load(model_path)


class MixtureModel(rv_continuous):
    def __init__(self, submodels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels

    def _pdf(self, x, **kwargs):
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def seed_all(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)