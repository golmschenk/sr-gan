"""Model architectures code."""
import torch
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu

from coefficient.data import irrelevant_data_multiplier
from utility import gpu, seed_all

observation_count = 10


class Generator(Module):
    """The generator model."""
    def __init__(self, hidden_size=10):
        super().__init__()
        self.input_size = 10
        self.linear1 = Linear(self.input_size, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, hidden_size)
        self.linear4 = Linear(hidden_size, observation_count * irrelevant_data_multiplier)

    def forward(self, x, add_noise=False):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x


class MLP(Module):
    """The DNN MLP model."""
    def __init__(self, hidden_size=10):
        super().__init__()
        seed_all(0)
        self.linear1 = Linear(observation_count * irrelevant_data_multiplier, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, hidden_size)
        self.linear4 = Linear(hidden_size, 1)
        self.features = None
        self.gradient_sum = torch.tensor(0, device=gpu)

    def forward(self, x):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        self.features = x
        x = self.linear4(x)
        return x.squeeze()


class DgganMLP(Module):
    """The DNN MLP model."""
    def __init__(self, hidden_size=10):
        super().__init__()
        seed_all(0)
        self.linear1 = Linear(observation_count * irrelevant_data_multiplier, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, hidden_size)
        self.linear4 = Linear(hidden_size, 2)
        self.features = None
        self.gradient_sum = torch.tensor(0, device=gpu)

    def forward(self, x):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        self.features = x
        x = self.linear4(x)
        return x[:, 0].squeeze(), x[:, 1].squeeze()


class SganMLP(Module):
    """The DNN MLP model for the SGAN."""
    def __init__(self, number_of_bins=10):
        super().__init__()
        seed_all(0)
        self.linear1 = Linear(observation_count * irrelevant_data_multiplier, 100)
        self.linear2 = Linear(100, 100)
        self.linear3 = Linear(100, 100)
        self.linear4 = Linear(100, number_of_bins)
        self.features = None
        self.gradient_sum = torch.tensor(0, device=gpu)

    def forward(self, x):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x.squeeze()
