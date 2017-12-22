"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import numpy as np
from scipy.stats import norm, gamma
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter as SummaryWriter_
import torch

import settings
from data import ToyDataset, MixtureModel
from presentation import generate_learning_process_images


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

    def add_histogram(self, tag, values, global_step=None, bins='auto'):
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_histogram(tag, values, global_step, bins)

def run_rsgan(steps):
    datetime_string = datetime.datetime.now().strftime('y%Ym%md%dh%Hm%Ms%S')
    trial_directory = os.path.join(settings.logs_directory, '{} {}'.format(settings.trial_name, datetime_string))
    os.makedirs(os.path.join(trial_directory, settings.temporary_directory))
    dnn_summary_writer = SummaryWriter(os.path.join(trial_directory, 'DNN'))
    gan_summary_writer = SummaryWriter(os.path.join(trial_directory, 'GAN'))
    dnn_summary_writer.summary_period = 100
    gan_summary_writer.summary_period = 100
    observation_count = 10
    noise_size = 10

    train_dataset = ToyDataset(dataset_size=100, observation_count=observation_count)
    train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

    unlabeled_dataset = ToyDataset(dataset_size=50000, observation_count=observation_count)
    unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True)

    test_dataset_size = 1000
    test_dataset = ToyDataset(test_dataset_size, observation_count)

    class Generator(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(noise_size, 50)
            self.linear5 = Linear(50, 30)
            self.linear6 = Linear(30, observation_count)

        def forward(self, x):
            x = leaky_relu(self.linear1(x))
            x = leaky_relu(self.linear5(x))
            x = self.linear6(x)
            return x

    class FakeGenerator(Module):
        def __init__(self):
            super().__init__()
            self.fake_parameters = Linear(1, 1)

        def forward(self, x):
            mean_model = MixtureModel([norm(0, 1)])
            std_model = MixtureModel([gamma(2)])
            means = mean_model.rvs(size=[x.size()[0], 1]).astype(dtype=np.float32)
            stds = std_model.rvs(size=[x.size()[0], 1]).astype(dtype=np.float32)
            fake_examples = np.random.normal(means, stds, size=[x.size()[0], observation_count]).astype(dtype=np.float32)
            fake_examples = Variable(torch.from_numpy(fake_examples))
            return fake_examples

    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(observation_count, 32)
            self.linear3 = Linear(32, 8)
            self.linear4 = Linear(8, 2)
            self.feature_layer = None
            self.gradient_sum = Variable(torch.zeros(1))
            self.register_gradient_sum_hooks()

        def forward(self, x):
            x = leaky_relu(self.linear1(x))
            x = leaky_relu(self.linear3(x))
            self.feature_layer = x
            x = self.linear4(x)
            return x

        def register_gradient_sum_hooks(self):
            def gradient_sum_hook(grad):
                nonlocal self
                self.gradient_sum += grad.abs().sum()
                return grad
            [parameter.register_hook(gradient_sum_hook) for parameter in self.parameters()]

        def zero_gradient_sum(self):
            self.gradient_sum = Variable(torch.zeros(1))


    G = FakeGenerator()
    D = MLP()
    DNN = MLP()
    d_lr = 1e-4
    g_lr = d_lr

    betas = (0.9, 0.999)
    D_optimizer = Adam(D.parameters(), lr=d_lr, betas=betas)
    G_optimizer = Adam(G.parameters(), lr=g_lr, betas=betas)
    DNN_optimizer = Adam(DNN.parameters(), lr=d_lr, betas=betas)

    all_fake_examples = None
    all_unlabeled_predictions = None
    all_test_predictions = None
    all_dnn_test_predictions = None
    all_train_predictions = None
    all_dnn_train_predictions = None
    step_time_start = datetime.datetime.now()

    for step in range(steps):
        labeled_examples, labels = next(iter(train_dataset_loader))
        # DNN.
        gan_summary_writer.step = step
        dnn_summary_writer.step = step
        if step % 500 == 0 and step != 0:
            print('Step {}, {}...'.format(step, datetime.datetime.now() - step_time_start))
            step_time_start = datetime.datetime.now()
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(Variable(labeled_examples))
        dnn_loss = torch.abs(dnn_predicted_labels[:, 0] - Variable(labels[:, 0])).pow(2).mean()
        dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.data[0])
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(Variable(labeled_examples))
        labeled_loss = torch.abs(predicted_labels[:, 0] - Variable(labels[:, 0])).pow(2).mean()
        gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.data[0])
        D.zero_gradient_sum()
        labeled_loss.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Labeled', D.gradient_sum.data[0])
        gan_summary_writer.add_histogram('Labeled', D.feature_layer.mean(0).data)
        # Unlabeled.
        _ = D(Variable(labeled_examples))
        labeled_feature_layer = D.feature_layer
        unlabeled_examples, _ = next(iter(unlabeled_dataset_loader))
        _ = D(Variable(unlabeled_examples))
        unlabeled_feature_layer = D.feature_layer
        unlabeled_loss = (unlabeled_feature_layer.mean(0) - labeled_feature_layer.mean(0)).pow(2).mean()
        gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.data[0])
        D.zero_gradient_sum()
        unlabeled_loss.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Unlabeled', D.gradient_sum.data[0])
        gan_summary_writer.add_histogram('Unlabeled', unlabeled_feature_layer.mean(0).data)
        # Fake.
        _ = D(Variable(unlabeled_examples))
        unlabeled_feature_layer = D.feature_layer
        z = torch.randn(settings.batch_size, noise_size)
        fake_examples = G(Variable(z))
        _ = D(fake_examples.detach())
        fake_feature_layer = D.feature_layer
        fake_loss = (unlabeled_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).sum().pow(1/2).neg()
        gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.data[0])
        D.zero_gradient_sum()
        fake_loss.backward()
        gan_summary_writer.add_histogram('Fake', fake_feature_layer.mean(0).data)
        gan_summary_writer.add_scalar('Gradient Sums/Fake', D.gradient_sum.data[0])
        # Gradient penalty.
        alpha = Variable(torch.rand(2, settings.batch_size, 1))
        alpha = alpha / alpha.sum(0)
        interpolates = (alpha[0] * Variable(unlabeled_examples, requires_grad=True) +
                        alpha[1] * Variable(fake_examples.detach().data, requires_grad=True))
        interpolates_predictions = D(interpolates)
        gradients = torch.autograd.grad(outputs=interpolates_predictions, inputs=interpolates,
                                        grad_outputs=torch.ones(interpolates_predictions.size()),
                                        create_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        D.zero_gradient_sum()
        gradient_penalty.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Gradient Penalty', D.gradient_sum.data[0])
        # Discriminator update.
        D_optimizer.step()
        # Generator.
        if step % 5 == 0:
            G_optimizer.zero_grad()
            _ = D(Variable(unlabeled_examples))
            unlabeled_feature_layer = D.feature_layer.detach()
            z = torch.randn(settings.batch_size, noise_size)
            fake_examples = G(Variable(z))
            _ = D(fake_examples)
            fake_feature_layer = D.feature_layer
            generator_loss = (unlabeled_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).mean()
            gan_summary_writer.add_scalar('Generator/Loss', generator_loss.data[0])
            generator_loss.backward()
            G_optimizer.step()

        if dnn_summary_writer.step % dnn_summary_writer.summary_period == 0:
            dnn_predicted_train_labels = DNN(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32)))).data.numpy()
            dnn_train_label_errors = np.mean(np.abs(dnn_predicted_train_labels - train_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Train Error/Mean', dnn_train_label_errors.data[0])
            dnn_summary_writer.add_scalar('Train Error/Std', dnn_train_label_errors.data[1])
            dnn_predicted_test_labels = DNN(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32)))).data.numpy()
            dnn_test_label_errors = np.mean(np.abs(dnn_predicted_test_labels - test_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Test Error/Mean', dnn_test_label_errors.data[0])
            dnn_summary_writer.add_scalar('Test Error/Std', dnn_test_label_errors.data[1])
            predicted_train_labels = D(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32)))).data.numpy()
            gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Train Error/Mean', gan_train_label_errors.data[0])
            gan_summary_writer.add_scalar('Train Error/Std', gan_train_label_errors.data[1])
            predicted_test_labels = D(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32)))).data.numpy()
            gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Test Error/Mean', gan_test_label_errors.data[0])
            gan_summary_writer.add_scalar('Test Error/Std', gan_test_label_errors.data[1])
            gan_summary_writer.add_scalar('Test Error/Ratio Mean GAN DNN', gan_test_label_errors.data[0] / dnn_test_label_errors.data[0])

            if dnn_summary_writer.step % 100 == 0:
                z = torch.randn(test_dataset_size, noise_size)
                fake_examples = G(Variable(z))
                fake_examples_array = fake_examples.data
                if all_fake_examples is None:
                    all_fake_examples = np.memmap(os.path.join(trial_directory, settings.temporary_directory, 'fake_examples.memmap'), dtype='float32', mode='w+',
                                                  shape=(1, *fake_examples_array.shape))
                    all_fake_examples[0] = fake_examples_array
                else:
                    all_fake_examples = np.append(all_fake_examples, fake_examples_array[np.newaxis], axis=0)
                unlabeled_examples_array = unlabeled_dataset.examples[:test_dataset_size]
                unlabeled_examples = torch.from_numpy(unlabeled_examples_array.astype(np.float32))
                unlabeled_predictions = D(Variable(unlabeled_examples))
                unlabeled_predictions_array = unlabeled_predictions.data
                if all_unlabeled_predictions is None:
                    all_unlabeled_predictions = np.memmap(os.path.join(trial_directory, settings.temporary_directory, 'unlabeled_predictions.memmap'), dtype='float32', mode='w+',
                                                          shape=(1, *unlabeled_predictions_array.shape))
                    all_unlabeled_predictions[0] = unlabeled_predictions_array
                else:
                    all_unlabeled_predictions = np.append(all_unlabeled_predictions,
                                                          unlabeled_predictions_array[np.newaxis], axis=0)
                test_predictions_array = predicted_test_labels
                if all_test_predictions is None:
                    all_test_predictions = np.memmap(os.path.join(trial_directory, settings.temporary_directory, 'test_predictions.memmap'), dtype='float32', mode='w+',
                                                          shape=(1, *test_predictions_array.shape))
                    all_test_predictions[0] = test_predictions_array
                else:
                    all_test_predictions = np.append(all_test_predictions,
                                                     test_predictions_array[np.newaxis], axis=0)
                train_predictions_array = predicted_train_labels
                if all_train_predictions is None:
                    all_train_predictions = np.memmap(
                        os.path.join(trial_directory, settings.temporary_directory, 'train_predictions.memmap'), dtype='float32',
                        mode='w+',
                        shape=(1, *train_predictions_array.shape))
                    all_train_predictions[0] = train_predictions_array
                else:
                    all_train_predictions = np.append(all_train_predictions,
                                                     train_predictions_array[np.newaxis], axis=0)
                dnn_test_predictions_array = dnn_predicted_test_labels
                if all_dnn_test_predictions is None:
                    all_dnn_test_predictions = np.memmap(
                        os.path.join(trial_directory, settings.temporary_directory, 'dnn_test_predictions.memmap'), dtype='float32', mode='w+',
                        shape=(1, *dnn_test_predictions_array.shape))
                    all_dnn_test_predictions[0] = dnn_test_predictions_array
                else:
                    all_dnn_test_predictions = np.append(all_dnn_test_predictions,
                                                          dnn_test_predictions_array[np.newaxis], axis=0)
                dnn_train_predictions_array = dnn_predicted_train_labels
                if all_dnn_train_predictions is None:
                    all_dnn_train_predictions = np.memmap(
                        os.path.join(trial_directory, settings.temporary_directory, 'dnn_train_predictions.memmap'), dtype='float32',
                        mode='w+',
                        shape=(1, *dnn_train_predictions_array.shape))
                    all_dnn_train_predictions[0] = dnn_train_predictions_array
                else:
                    all_dnn_train_predictions = np.append(all_dnn_train_predictions,
                                                         dnn_train_predictions_array[np.newaxis], axis=0)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'fake_examples.npy'), all_fake_examples)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'unlabeled_predictions.npy'), all_unlabeled_predictions)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'test_predictions.npy'), all_test_predictions)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'dnn_test_predictions.npy'), all_dnn_test_predictions)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'train_predictions.npy'), all_train_predictions)
    np.save(os.path.join(trial_directory, settings.temporary_directory, 'dnn_train_predictions.npy'), all_dnn_train_predictions)

    predicted_train_labels = DNN(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32)))).data.numpy()
    dnn_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
    predicted_test_labels = DNN(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32)))).data.numpy()
    dnn_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)

    predicted_train_labels = D(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32)))).data.numpy()
    gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
    predicted_test_labels = D(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32)))).data.numpy()
    gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)

    generate_learning_process_images(trial_directory)
    print('Completed {}'.format(trial_directory))

    return dnn_train_label_errors, dnn_test_label_errors, gan_train_label_errors, gan_test_label_errors


for steps in [settings.steps_to_run]:
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

