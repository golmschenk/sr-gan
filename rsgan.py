"""
Regression semi-supervised GAN code.
"""

import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam
import torch


def run_rsgan(steps):
    observation_count = 100
    noise_size = 10

    train_dataset_size = 10000
    train_means = np.random.normal(size=[train_dataset_size, 1])
    train_standard_deviations = np.random.gamma(shape=2, size=[train_dataset_size, 1])
    train_examples = np.random.normal(train_means, train_standard_deviations, size=[train_dataset_size, observation_count])
    train_examples.sort(axis=1)
    train_labels = np.concatenate((train_means, train_standard_deviations), axis=1)

    test_dataset_size = 1000
    test_means = np.random.normal(size=[test_dataset_size, 1])
    test_standard_deviations = np.random.gamma(shape=2, size=[test_dataset_size, 1])
    test_examples = np.random.normal(test_means, test_standard_deviations, size=[test_dataset_size, observation_count])
    test_examples.sort(axis=1)
    test_labels = np.concatenate((test_means, test_standard_deviations), axis=1)


    class Generator(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(noise_size, 16)
            self.linear2 = Linear(16, 32)
            self.linear3 = Linear(32, observation_count)

        def forward(self, x):
            x = leaky_relu(self.linear1(x))
            x = leaky_relu(self.linear2(x))
            x = self.linear3(x)
            return x


    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(observation_count, 32)
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
    DNN = MLP()
    D_optimizer = Adam(D.parameters())
    G_optimizer = Adam(G.parameters())
    DNN_optimizer = Adam(DNN.parameters())

    for step in range(steps):
        # DNN.
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(Variable(torch.from_numpy(train_examples.astype(np.float32))))
        dnn_loss = torch.abs(dnn_predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32))))
        labeled_feature_layer = D.feature_layer.detach()
        labeled_loss = torch.abs(predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
        labeled_loss.backward()
        # Unlabeled.
        unlabeled_means = np.random.normal(size=[train_dataset_size, 1])
        unlabeled_standard_deviations = np.random.gamma(shape=2, size=[train_dataset_size, 1])
        unlabeled_examples = np.random.normal(unlabeled_means, unlabeled_standard_deviations, size=[train_dataset_size, observation_count])
        _ = D(Variable(torch.from_numpy(unlabeled_examples.astype(np.float32))))
        unlabeled_feature_layer = D.feature_layer
        unlabeled_loss = torch.abs(unlabeled_feature_layer.mean(0) - labeled_feature_layer.mean(0)).mean()
        unlabeled_loss.backward()
        unlabeled_feature_layer = unlabeled_feature_layer.detach()
        # Fake.
        z = torch.randn(train_dataset_size, noise_size)
        fake_examples = G(Variable(z))
        _ = D(fake_examples)
        fake_feature_layer = D.feature_layer
        real_feature_layer = (labeled_feature_layer + unlabeled_feature_layer) / 2
        fake_loss = torch.abs(real_feature_layer.mean(0) - fake_feature_layer.mean(0)).log().mean().neg()
        fake_loss.backward()
        # Discriminator update.
        D_optimizer.step()
        # Generator.
        G_optimizer.zero_grad()
        z = torch.randn(train_dataset_size, noise_size)
        fake_examples = G(Variable(z))
        _ = D(fake_examples)
        fake_feature_layer = D.feature_layer
        real_feature_layer = (labeled_feature_layer + unlabeled_feature_layer) / 2
        generator_loss = torch.abs(real_feature_layer.mean(0) - fake_feature_layer.mean(0)).mean()
        generator_loss.backward()
        G_optimizer.step()

    predicted_train_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
    train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)

    simple_predicted_means = train_examples.mean(axis=1)
    simple_mean_errors = np.mean(np.abs(simple_predicted_means - train_labels[:, 0]))

    print('Simple: {}'.format(simple_mean_errors))
    print('Model:  {}'.format(train_label_errors[0]))

    predicted_test_labels = D(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)
    return train_label_errors, test_label_errors


for steps in [1000, 5000, 10000]:
    set_train_losses = []
    set_test_losses = []
    for index in range(5):
        train_loss_, test_loss_ = run_rsgan(steps)
        set_train_losses.append(train_loss_)
        set_test_losses.append(test_loss_)
    print('Steps: {}'.format(steps))
    print('Train: {}'.format(np.mean(set_train_losses, axis=0)))
    print('Test: {}'.format(np.mean(set_test_losses, axis=0)))
    print()

# losses = []
# for _ in range(1):
#     losses.append(run_rsgan())
# print('Mean RSGAN loss: {}'.format(sum(losses) / len(losses)))

