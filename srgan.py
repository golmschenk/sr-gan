"""
Regression semi-supervised GAN code.
"""
import datetime
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
    datetime_string = datetime.datetime.now().strftime('y%Ym%md%dh%Hm%Ms%S')
    trial_name = 'e1000 o10 fur1 lr1e-5'
    dnn_summary_writer = SummaryWriter('logs/dnn {} {}'.format(trial_name, datetime_string))
    gan_summary_writer = SummaryWriter('logs/gan {} {}'.format(trial_name, datetime_string))
    dnn_summary_writer.summary_period = 10
    gan_summary_writer.summary_period = 10
    observation_count = 10
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
    d_lr = 1e-5
    g_lr = d_lr

    betas = (0.5, 0.9)
    D_optimizer = Adam(D.parameters(), lr=d_lr, betas=betas)
    G_optimizer = Adam(G.parameters(), lr=g_lr, betas=betas)
    DNN_optimizer = Adam(DNN.parameters(), lr=d_lr, betas=betas)

    for step in range(steps):
        labeled_examples = torch.from_numpy(train_examples.astype(np.float32))
        labels = torch.from_numpy(train_labels.astype(np.float32))
        # DNN.
        gan_summary_writer.step = step
        dnn_summary_writer.step = step
        if step % 500 == 0 and step != 0:
            print('Step {}...'.format(step))
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(Variable(labeled_examples))
        dnn_loss = torch.abs(dnn_predicted_labels - Variable(labels)).pow(2).mean()
        dnn_summary_writer.add_scalar('Labeled Loss', dnn_loss.data[0])
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(Variable(labeled_examples))
        detached_labeled_feature_layer = D.feature_layer.detach()
        labeled_loss = torch.abs(predicted_labels - Variable(labels)).pow(2).mean()
        gan_summary_writer.add_scalar('Labeled Loss', labeled_loss.data[0])
        labeled_loss.backward()
        # Unlabeled.
        unlabeled_means = np.random.normal(size=[train_dataset_size, 1])
        unlabeled_standard_deviations = np.random.gamma(shape=2, size=[train_dataset_size, 1])
        unlabeled_examples_array = np.random.normal(unlabeled_means, unlabeled_standard_deviations, size=[train_dataset_size, observation_count])
        unlabeled_examples = torch.from_numpy(unlabeled_examples_array.astype(np.float32))
        _ = D(Variable(unlabeled_examples))
        unlabeled_feature_layer = D.feature_layer
        detached_unlabeled_feature_layer = unlabeled_feature_layer.detach()
        unlabeled_loss = (unlabeled_feature_layer.mean(0) - detached_labeled_feature_layer.mean(0)).pow(2).mean()
        gan_summary_writer.add_scalar('Unlabeled Loss', unlabeled_loss.data[0])
        unlabeled_loss.backward()
        # Fake.
        z = torch.randn(train_dataset_size, noise_size)
        fake_examples = G(Variable(z))
        _ = D(fake_examples.detach())
        fake_feature_layer = D.feature_layer
        real_feature_layer = (detached_labeled_feature_layer + detached_unlabeled_feature_layer) / 2
        fake_loss = ((real_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2) + 1).log().mean().neg()
        gan_summary_writer.add_scalar('Fake Loss', fake_loss.data[0])
        fake_loss.backward()
        # Gradient penalty.
        alpha = Variable(torch.rand(3, train_dataset_size, 1))
        alpha = alpha / alpha.sum(0)
        interpolates = alpha[0] * Variable(labeled_examples, requires_grad=True) + alpha[1] * Variable(unlabeled_examples, requires_grad=True) + alpha[2] * Variable(fake_examples.detach().data, requires_grad=True)
        interpolates_predictions = D(interpolates)
        gradients = torch.autograd.grad(outputs=interpolates_predictions, inputs=interpolates,
                                        grad_outputs=torch.ones(interpolates_predictions.size()),
                                        create_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        gradient_penalty.backward()
        # Discriminator update.
        D_optimizer.step()
        # Generator.
        if step % 5 == 0:
            G_optimizer.zero_grad()
            z = torch.randn(train_dataset_size, noise_size)
            fake_examples = G(Variable(z))
            _ = D(fake_examples)
            fake_feature_layer = D.feature_layer
            real_feature_layer = (detached_labeled_feature_layer + detached_unlabeled_feature_layer) / 2
            generator_loss = (real_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).mean()
            gan_summary_writer.add_scalar('Generator Loss', generator_loss.data[0])
            generator_loss.backward()
            G_optimizer.step()

        if dnn_summary_writer.step % dnn_summary_writer.summary_period == 0:
            predicted_train_labels = DNN(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
            dnn_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
            dnn_summary_writer.add_scalar('Train Error Mean', dnn_train_label_errors.data[0])
            dnn_summary_writer.add_scalar('Train Error Std', dnn_train_label_errors.data[1])
            predicted_test_labels = DNN(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
            dnn_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)
            dnn_summary_writer.add_scalar('Test Error Mean', dnn_test_label_errors.data[0])
            dnn_summary_writer.add_scalar('Test Error Std', dnn_test_label_errors.data[1])
            predicted_train_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
            gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
            gan_summary_writer.add_scalar('Train Error Mean', gan_train_label_errors.data[0])
            gan_summary_writer.add_scalar('Train Error Std', gan_train_label_errors.data[1])
            predicted_test_labels = D(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
            gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)
            gan_summary_writer.add_scalar('Test Error Mean', gan_test_label_errors.data[0])
            gan_summary_writer.add_scalar('Test Error Std', gan_test_label_errors.data[1])

    predicted_train_labels = DNN(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
    dnn_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
    predicted_test_labels = DNN(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    dnn_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)

    predicted_train_labels = D(Variable(torch.from_numpy(train_examples.astype(np.float32)))).data.numpy()
    gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_labels), axis=0)
    predicted_test_labels = D(Variable(torch.from_numpy(test_examples.astype(np.float32)))).data.numpy()
    gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_labels), axis=0)

    return dnn_train_label_errors, dnn_test_label_errors, gan_train_label_errors, gan_test_label_errors


for steps in [500000]:
    set_gan_train_losses = []
    set_gan_test_losses = []
    set_dnn_train_losses = []
    set_dnn_test_losses = []
    for index in range(1):
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

