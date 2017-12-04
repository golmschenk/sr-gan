"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam, RMSprop
from tensorboardX import SummaryWriter as SummaryWriter_
import torch


class SummaryWriter(SummaryWriter_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.summary_period = 1

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_scalar(tag, scalar_value, global_step)


def run_rsgan(steps):
    datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
    dnn_summary_writer = SummaryWriter('logs/dnn {}'.format(datetime_string))
    gan_summary_writer = SummaryWriter('logs/gan {}'.format(datetime_string))
    dnn_summary_writer.summary_period = 10
    gan_summary_writer.summary_period = 10
    observation_count = 100
    noise_size = 10

    train_dataset_size = 1000
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
    D_optimizer = RMSprop(D.parameters())
    G_optimizer = RMSprop(G.parameters())
    DNN_optimizer = RMSprop(DNN.parameters())

    for step in range(steps):
        # DNN.
        gan_summary_writer.step = step
        dnn_summary_writer.step = step
        if step % 500 == 0 and step != 0:
            print('Step {}...'.format(step))
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(Variable(torch.from_numpy(train_examples.astype(np.float32))))
        dnn_loss = torch.abs(dnn_predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
        dnn_summary_writer.add_scalar('Labeled Loss', dnn_loss.data[0])
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32))))
        labeled_feature_layer = D.feature_layer.detach()
        labeled_loss = torch.abs(predicted_labels - Variable(torch.from_numpy(train_labels.astype(np.float32)))).mean()
        gan_summary_writer.add_scalar('Labeled Loss', labeled_loss.data[0])
        labeled_loss.backward()
        # Unlabeled.
        unlabeled_means = np.random.normal(size=[train_dataset_size, 1])
        unlabeled_standard_deviations = np.random.gamma(shape=2, size=[train_dataset_size, 1])
        unlabeled_examples = np.random.normal(unlabeled_means, unlabeled_standard_deviations, size=[train_dataset_size, observation_count])
        _ = D(Variable(torch.from_numpy(unlabeled_examples.astype(np.float32))))
        unlabeled_feature_layer = D.feature_layer
        unlabeled_loss = (unlabeled_feature_layer.mean(0) - labeled_feature_layer.mean(0)).pow(2).mean()
        gan_summary_writer.add_scalar('Unlabeled Loss', unlabeled_loss.data[0])
        unlabeled_loss.backward()
        unlabeled_feature_layer = unlabeled_feature_layer.detach()
        # Fake.
        z = torch.randn(train_dataset_size, noise_size)
        fake_examples = G(Variable(z))
        _ = D(fake_examples)
        fake_feature_layer = D.feature_layer
        real_feature_layer = (labeled_feature_layer + unlabeled_feature_layer) / 2
        fake_loss = ((real_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2) + 1).log().mean().neg()
        gan_summary_writer.add_scalar('Fake Loss', fake_loss.data[0])
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
        generator_loss = (real_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).mean()
        gan_summary_writer.add_scalar('Generator Loss', generator_loss.data[0])
        generator_loss.backward()
        G_optimizer.step()

    predicted_train_labels = DNN(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
    dnn_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
    predicted_test_labels = DNN(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    dnn_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)

    predicted_train_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
    gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
    predicted_test_labels = D(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)

    return dnn_train_label_errors, dnn_test_label_errors, gan_train_label_errors, gan_test_label_errors


for steps in [1000, 5000]:
    set_gan_train_losses = []
    set_gan_test_losses = []
    set_dnn_train_losses = []
    set_dnn_test_losses = []
    for index in range(3):
        print('Running trial number {}...'.format(index))
        dnn_train_label_errors_, dnn_test_label_errors_, gan_train_label_errors_, gan_test_label_errors_ = run_rsgan(steps)
        set_dnn_train_losses.append(dnn_train_label_errors_)
        set_dnn_test_losses.append(dnn_test_label_errors_)
        set_gan_train_losses.append(gan_train_label_errors_)
        set_gan_test_losses.append(gan_test_label_errors_)
    print('Steps: {}'.format(steps))
    print('DNN Train: {}'.format(np.mean(set_dnn_train_losses, axis=0)))
    print('DNN Test: {}'.format(np.mean(set_dnn_test_losses, axis=0)))
    print('GAN Train: {}'.format(np.mean(set_gan_train_losses, axis=0)))
    print('GAN Test: {}'.format(np.mean(set_gan_test_losses, axis=0)))
    print()

# losses = []
# for _ in range(1):
#     losses.append(run_rsgan())
# print('Mean RSGAN loss: {}'.format(sum(losses) / len(losses)))

