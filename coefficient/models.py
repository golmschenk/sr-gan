import torch
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu

from coefficient.data import irrelevant_data_multiplier
from utility import gpu, seed_all

observation_count = 10


class Generator(Module):
    """The generator model."""

    def __init__(self):
        super().__init__()
        self.input_size = 10
        self.linear1 = Linear(self.input_size, 100)
        self.linear2 = Linear(100, 100)
        self.linear3 = Linear(100, 100)
        self.linear4 = Linear(100, observation_count * irrelevant_data_multiplier)

    def forward(self, x, add_noise=False):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x


class MLP(Module):
    """The DNN MLP model."""

    def __init__(self):
        super().__init__()
        seed_all(0)
        self.linear1 = Linear(observation_count * irrelevant_data_multiplier, 100)
        self.linear2 = Linear(100, 100)
        self.linear3 = Linear(100, 100)
        self.linear4 = Linear(100, 1)
        self.feature_layer = None
        self.gradient_sum = torch.tensor(0, device=gpu)

    def forward(self, x, add_noise=False):
        """The forward pass of the module."""
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        self.feature_layer = x
        x = self.linear4(x)
        return x.squeeze()

    def register_gradient_sum_hooks(self):
        """A hook to remember the sum gradients of a backwards call."""

        def gradient_sum_hook(grad):
            """The hook callback."""
            nonlocal self
            self.gradient_sum += grad.abs().sum()
            return grad

        [parameter.register_hook(gradient_sum_hook) for parameter in self.parameters()]

    def zero_gradient_sum(self):
        """Zeros the sum gradients to allow for a new summing for logging."""
        self.gradient_sum = torch.tensor(0, device=gpu)