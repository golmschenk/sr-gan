import random
import re

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