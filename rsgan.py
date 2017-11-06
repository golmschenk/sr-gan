"""
Regression semi-supervised GAN code.
"""

import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam
import torch


def run_rsgan():
    train_batch_size = 100
    train_means = np.random.normal(size=[train_batch_size, 1])
    train_standard_deviations = np.random.gamma(shape=2, size=[train_batch_size, 1])
    train_examples = np.random.normal(train_means, train_standard_deviations, size=[train_batch_size, 10])
    train_labels = np.concatenate((train_means, train_standard_deviations), axis=1)

    test_batch_size = 1000
    test_means = np.random.normal(size=[test_batch_size, 1])
    test_standard_deviations = np.random.gamma(shape=2, size=[test_batch_size, 1])
    test_examples = np.random.normal(test_means, test_standard_deviations, size=[test_batch_size, 10])
    test_labels = np.concatenate((test_means, test_standard_deviations), axis=1)


    class Generator(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(10, 16)
            self.linear2 = Linear(16, 32)
            self.linear3 = Linear(32, 10)

        def forward(self, x):
            x = leaky_relu(self.linear1(x))
            x = leaky_relu(self.linear2(x))
            x = self.linear3(x)
            return x


    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(10, 32)
            self.linear2 = Linear(32, 16)
            self.linear3 = Linear(16, 2)
            self.feature_layer = None

        def forward(self, x):
            x = leaky_relu(self.linear1(x))
            x = leaky_relu(self.linear2(x))
            self.feature_layer = x
            x = self.linear3(x)
            return x


    G = Generator()
    D = MLP()
    D_optimizer = Adam(D.parameters())
    G_optimizer = Adam(G.parameters())

    for step in range(10000):
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32))))
        labeled_feature_layer = D.feature_layer.detach()
        labeled_loss = torch.abs(predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
        labeled_loss.backward()
        # Unlabeled.
        unlabeled_means = np.random.normal(size=[train_batch_size, 1])
        unlabeled_standard_deviations = np.random.gamma(shape=2, size=[train_batch_size, 1])
        unlabeled_examples = np.random.normal(unlabeled_means, unlabeled_standard_deviations, size=[train_batch_size, 10])
        _ = D(Variable(torch.from_numpy(unlabeled_examples.astype(np.float32))))
        unlabeled_feature_layer = D.feature_layer
        unlabeled_loss = torch.abs(unlabeled_feature_layer - labeled_feature_layer).mean()
        #unlabeled_loss.backward()
        unlabeled_feature_layer = unlabeled_feature_layer.detach()
        # Fake.
        z = torch.randn(train_batch_size, 10)
        fake_examples = G(Variable(z))
        _ = D(fake_examples)
        fake_feature_layer = D.feature_layer
        real_feature_layer = (labeled_feature_layer + unlabeled_feature_layer) / 2
        fake_loss = torch.abs(real_feature_layer.mean(0) - fake_feature_layer.mean(0)).log().mean().neg()
        #fake_loss.backward()
        # Discriminator update.
        D_optimizer.step()
        # Generator.
        G_optimizer.zero_grad()
        z = torch.randn(train_batch_size, 10)
        fake_examples = G(Variable(z))
        _ = D(fake_examples)
        fake_feature_layer = D.feature_layer
        real_feature_layer = (labeled_feature_layer + unlabeled_feature_layer) / 2
        generator_loss = torch.abs(real_feature_layer.mean(0) - fake_feature_layer.mean(0)).mean()
        #generator_loss.backward()
        G_optimizer.step()

    predicted_labels = D(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    test_loss = np.mean(np.abs(predicted_labels - test_labels))
    return test_loss


losses = []
for _ in range(10):
    losses.append(run_rsgan())
print('Mean RSGAN loss: {}'.format(sum(losses) / len(losses)))

