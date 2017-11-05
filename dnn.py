"""
Code for a basic regression deep neural network.
"""

import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam
import torch


train_means = np.random.normal(size=[1000, 1])
train_standard_deviations = np.random.gamma(shape=2, size=[1000, 1])
train_examples = np.random.normal(train_means, train_standard_deviations, size=[1000, 10])
train_labels = np.concatenate((train_means, train_standard_deviations), axis=1)

test_means = np.random.normal(size=[100, 1])
test_standard_deviations = np.random.gamma(shape=2, size=[100, 1])
test_examples = np.random.normal(test_means, test_standard_deviations, size=[100, 10])
test_labels = np.concatenate((test_means, test_standard_deviations), axis=1)

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10, 32)
        self.linear2 = Linear(32, 16)
        self.linear3 = Linear(16, 2)

    def forward(self, x):
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

net = MLP()
optimizer = Adam(net.parameters())

for step in range(1000):
    optimizer.zero_grad()
    predicted_labels = net(Variable(torch.from_numpy(train_examples.astype(np.float32))))
    loss = torch.abs(predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
    loss.backward()
    optimizer.step()

predicted_labels = net(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
test_loss = np.mean(np.abs(predicted_labels - test_labels))
print(test_loss)
