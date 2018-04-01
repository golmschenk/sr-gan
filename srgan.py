"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import numpy as np
from scipy.stats import norm, wasserstein_distance
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.nn.functional import leaky_relu
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter as SummaryWriter_
import torch
import re

from settings import Settings
from data import ToyDataset, MixtureModel, irrelevant_data_multiplier, seed_all
from hardware import gpu, cpu
from presentation import generate_display_frame

global_trial_directory = None

seed_all(0)


def infinite_iter(dataset):
    """Create an infinite generator from a dataset."""
    while True:
        for examples in dataset:
            yield examples


def unit_vector(vector):
    """Gets the unit vector version of a vector."""
    return vector.div(vector.norm() + 1e-10)


def angle_between(vector0, vector1):
    """Calculates the angle between two vectors."""
    unit_vector0 = unit_vector(vector0)
    unit_vector1 = unit_vector(vector1)
    epsilon = 1e-6
    return unit_vector0.dot(unit_vector1).clamp(-1.0 + epsilon, 1.0 - epsilon).acos()


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
        if not settings_.histogram_logging:
            return
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


def coefficient_estimate_loss(predicted_labels, labels, order=2):
    """Calculate the loss from the coefficient prediction."""
    return (predicted_labels[:, 0] - gpu(Variable(labels[:, 0]))).abs().pow(2).sum().pow(1/2).pow(order)


def feature_distance_loss(base_features, other_features, order=2, base_noise=0, scale=False):
    """Calculate the loss based on the distance between feature vectors."""
    base_mean_features = base_features.mean(0)
    other_mean_features = other_features.mean(0)
    if base_noise:
        base_mean_features += torch.normal(torch.zeros_like(base_mean_features), base_mean_features * base_noise)
    mean_feature_distance = (base_mean_features - other_mean_features).abs().pow(2).sum().pow(1 / 2)
    if scale:
        epsilon = 1e-10
        mean_feature_distance /= (base_mean_features.norm() + other_mean_features.norm() + epsilon)
    return mean_feature_distance.pow(order)


def feature_angle_loss(base_features, other_features, target=0, summary_writer=None):
    """Calculate the loss based on the angle between feature vectors."""
    angle = angle_between(base_features.mean(0), other_features.mean(0))
    if summary_writer:
        summary_writer.add_scalar('Feature Vector/Angle', angle.data[0])
    return (angle - target).abs().pow(2)


def feature_corrcoef(x):
    """Calculate the feature vector's correlation coefficients."""
    transposed_x = x.transpose(0, 1)
    return corrcoef(transposed_x)


def corrcoef(x):
    """Calculate the correlation coefficients."""
    mean_x = x.mean(1, keepdim=True)
    xm = x.sub(mean_x)
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def feature_covariance_loss(base_features, other_features):
    """Calculate the loss between feature vector correlation coefficient distances."""
    base_corrcoef = feature_corrcoef(base_features)
    other_corrcoef = feature_corrcoef(other_features)
    return (base_corrcoef - other_corrcoef).abs().sum()


def run_srgan(settings):
    """
    Train the SRGAN

    :param settings: The settings object.
    :type settings: Settings
    """
    datetime_string = datetime.datetime.now().strftime('y%Ym%md%dh%Hm%Ms%S')
    trial_directory = os.path.join(settings.logs_directory, '{} {}'.format(settings.trial_name, datetime_string))
    global global_trial_directory
    global_trial_directory = trial_directory
    os.makedirs(os.path.join(trial_directory, settings.temporary_directory))
    dnn_summary_writer = SummaryWriter(os.path.join(trial_directory, 'DNN'))
    gan_summary_writer = SummaryWriter(os.path.join(trial_directory, 'GAN'))
    dnn_summary_writer.summary_period = settings.summary_step_period
    gan_summary_writer.summary_period = settings.summary_step_period
    observation_count = 10
    noise_size = 10

    train_dataset = ToyDataset(dataset_size=settings.labeled_dataset_size, observation_count=observation_count, seed=3)
    train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

    unlabeled_dataset = ToyDataset(dataset_size=settings.unlabeled_dataset_size, observation_count=observation_count,
                                   seed=1)
    unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True)

    test_dataset = ToyDataset(settings.test_dataset_size, observation_count, seed=2)

    def add_layer_noise(add_noise, x):
        """Add noise to the given layer."""
        if add_noise:
            x += torch.normal(torch.zeros_like(x),
                              x.detach().norm(dim=1, keepdim=True).expand_as(x) * settings.noise_scale)
        return x

    class Generator(Module):
        """The generator model."""
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(noise_size, 20)
            self.linear5 = Linear(20, 30)
            self.linear6 = Linear(30, observation_count * irrelevant_data_multiplier)

        def forward(self, x, add_noise=False):
            """The forward pass of the module."""
            x = leaky_relu(self.linear1(x))
            x = add_layer_noise(add_noise, x)
            x = leaky_relu(self.linear5(x))
            x = add_layer_noise(add_noise, x)
            x = self.linear6(x)
            return x

    class MLP(Module):
        """The DNN MLP model."""
        def __init__(self):
            super().__init__()
            seed_all(0)
            self.linear1 = Linear(observation_count * irrelevant_data_multiplier, 16)
            self.linear3 = Linear(16, 4)
            self.linear4 = Linear(4, 1)
            self.feature_layer = None
            self.gradient_sum = gpu(Variable(torch.zeros(1)))

        def forward(self, x, add_noise=False):
            """The forward pass of the module."""
            x = add_layer_noise(add_noise, x)
            x = leaky_relu(self.linear1(x))
            x = add_layer_noise(add_noise, x)
            x = leaky_relu(self.linear3(x))
            x = add_layer_noise(add_noise, x)
            self.feature_layer = x
            x = self.linear4(x)
            return x

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
            self.gradient_sum = gpu(Variable(torch.zeros(1)))

    G_model = gpu(Generator())
    D_mlp = MLP()
    DNN_mlp = MLP()
    if settings.DNN_load_model_path:
        DNN_mlp.load_state_dict(torch.load(settings.DNN_load_model_path))
    if settings.D_load_model_path:
        D_mlp.load_state_dict(torch.load(settings.D_load_model_path))
    if settings.G_load_model_path:
        G_model.load_state_dict(torch.load(settings.G_load_model_path))
    G = gpu(G_model)
    D = gpu(D_mlp)
    DNN = gpu(DNN_mlp)
    d_lr = settings.learning_rate
    g_lr = d_lr

    # betas = (0.9, 0.999)
    weight_decay = 1e-2
    D_optimizer = Adam(D.parameters(), lr=d_lr, weight_decay=weight_decay)
    G_optimizer = Adam(G.parameters(), lr=g_lr)
    DNN_optimizer = Adam(DNN.parameters(), lr=d_lr, weight_decay=weight_decay)

    step_time_start = datetime.datetime.now()
    print(trial_directory)
    train_dataset_generator = infinite_iter(train_dataset_loader)
    unlabeled_dataset_generator = infinite_iter(unlabeled_dataset_loader)

    for step in range(settings.steps_to_run):
        labeled_examples, labels = next(train_dataset_generator)
        # DNN.
        gan_summary_writer.step = step
        dnn_summary_writer.step = step
        # f_summary_writer.step = step
        if step % settings.summary_step_period == 0 and step != 0:
            print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
            step_time_start = datetime.datetime.now()
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(gpu(Variable(labeled_examples)))
        dnn_loss = coefficient_estimate_loss(dnn_predicted_labels, labels) * settings.labeled_loss_multiplier
        dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.data[0])
        dnn_feature_layer = DNN.feature_layer
        dnn_summary_writer.add_scalar('Feature Norm/Labeled',
                                      float(np.linalg.norm(cpu(dnn_feature_layer.mean(0)).data.numpy(), ord=2)))
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(gpu(Variable(labeled_examples)))
        labeled_loss = coefficient_estimate_loss(predicted_labels, labels) * settings.labeled_loss_multiplier
        gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.data[0])
        D.zero_gradient_sum()
        labeled_loss.backward()
        # Unlabeled.
        _ = D(gpu(Variable(labeled_examples)))
        labeled_feature_layer = D.feature_layer
        gan_summary_writer.add_scalar('Feature Norm/Labeled',
                                      float(np.linalg.norm(cpu(labeled_feature_layer.mean(0)).data.numpy(), ord=2)))
        gan_summary_writer.add_histogram('Features/Labeled', cpu(labeled_feature_layer).data.numpy())
        unlabeled_examples, _ = next(unlabeled_dataset_generator)
        _ = D(gpu(Variable(unlabeled_examples)))
        unlabeled_feature_layer = D.feature_layer
        gan_summary_writer.add_histogram('Features/Unlabeled', cpu(unlabeled_feature_layer).data.numpy())
        unlabeled_loss = feature_distance_loss(unlabeled_feature_layer,
                                               labeled_feature_layer, scale=False) * settings.unlabeled_loss_multiplier
        gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.data[0])
        D.zero_gradient_sum()
        unlabeled_loss.backward()
        # Fake.
        _ = D(gpu(Variable(unlabeled_examples)))
        unlabeled_feature_layer = D.feature_layer
        z = torch.from_numpy(MixtureModel([norm(-settings.mean_offset, 1),
                                           norm(settings.mean_offset, 1)]
                                          ).rvs(size=[settings.batch_size, noise_size]).astype(np.float32))
        fake_examples = G(gpu(Variable(z)), add_noise=False)
        _ = D(fake_examples.detach())
        fake_feature_layer = D.feature_layer
        gan_summary_writer.add_histogram('Features/Fake', cpu(fake_feature_layer).data.numpy())
        fake_loss = feature_distance_loss(unlabeled_feature_layer, fake_feature_layer, scale=False,
                                          order=settings.fake_loss_order).neg() * settings.fake_loss_multiplier
        gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.data[0])
        D.zero_gradient_sum()
        fake_loss.backward()
        # Feature norm loss.
        _ = D(gpu(Variable(unlabeled_examples)))
        unlabeled_feature_layer = D.feature_layer
        feature_norm_loss = (unlabeled_feature_layer.norm(0).mean() - 1).pow(2)
        feature_norm_loss.backward()
        # Gradient penalty.
        if settings.gradient_penalty_on:
            alpha = gpu(Variable(torch.rand(2)))
            alpha = alpha / alpha.sum(0)
            interpolates = (alpha[0] * gpu(Variable(unlabeled_examples, requires_grad=True)) +
                            alpha[1] * gpu(Variable(fake_examples.detach().data, requires_grad=True)))
            _ = D(interpolates)
            interpolates_predictions = D.feature_layer
            gradients = torch.autograd.grad(outputs=interpolates_predictions, inputs=interpolates,
                                            grad_outputs=gpu(torch.ones(interpolates_predictions.size())),
                                            create_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * settings.gradient_penalty_multiplier
            D.zero_gradient_sum()
            gradient_penalty.backward()
        # Discriminator update.
        D_optimizer.step()
        # Generator.
        if step % settings.generator_training_step_period == 0:
            G_optimizer.zero_grad()
            _ = D(gpu(Variable(unlabeled_examples)), add_noise=False)
            unlabeled_feature_layer = D.feature_layer.detach()
            z = torch.randn(settings.batch_size, noise_size)
            fake_examples = G(gpu(Variable(z)))
            _ = D(fake_examples)
            fake_feature_layer = D.feature_layer
            generator_loss = feature_distance_loss(unlabeled_feature_layer, fake_feature_layer)
            gan_summary_writer.add_scalar('Generator/Loss', generator_loss.data[0])
            generator_loss.backward()
            G_optimizer.step()

        if dnn_summary_writer.step % dnn_summary_writer.summary_period == 0 or dnn_summary_writer.step % settings.presentation_step_period == 0:
            dnn_predicted_train_labels = cpu(DNN(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
            dnn_train_label_errors = np.mean(np.abs(dnn_predicted_train_labels - train_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Train Error/MAE', dnn_train_label_errors.data[0])
            dnn_predicted_test_labels = cpu(DNN(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
            dnn_test_label_errors = np.mean(np.abs(dnn_predicted_test_labels - test_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Test Error/MAE', dnn_test_label_errors.data[0])

            predicted_train_labels = cpu(D(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
            gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Train Error/MAE', gan_train_label_errors.data[0])
            predicted_test_labels = cpu(D(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
            gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Test Error/MAE', gan_test_label_errors.data[0])
            gan_summary_writer.add_scalar('Test Error/Ratio MAE GAN DNN', gan_test_label_errors.data[0] / dnn_test_label_errors.data[0])

            z = torch.from_numpy(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]).rvs(size=[settings.batch_size, noise_size]).astype(np.float32))
            fake_examples = G(gpu(Variable(z)), add_noise=False)
            fake_examples_array = cpu(fake_examples.data).numpy()
            fake_labels_array = np.mean(fake_examples_array, axis=1)
            unlabeled_labels_array = unlabeled_dataset.labels[:settings.test_dataset_size][:, 0]
            label_wasserstein_distance = wasserstein_distance(fake_labels_array, unlabeled_labels_array)
            gan_summary_writer.add_scalar('Generator/Label Wasserstein', label_wasserstein_distance)

            unlabeled_examples_array = unlabeled_dataset.examples[:settings.test_dataset_size]
            unlabeled_examples = torch.from_numpy(unlabeled_examples_array.astype(np.float32))
            unlabeled_predictions = D(gpu(Variable(unlabeled_examples)))

            if dnn_summary_writer.step % settings.presentation_step_period == 0:
                unlabeled_predictions_array = cpu(unlabeled_predictions.data).numpy()
                test_predictions_array = predicted_test_labels
                train_predictions_array = predicted_train_labels
                dnn_test_predictions_array = dnn_predicted_test_labels
                dnn_train_predictions_array = dnn_predicted_train_labels
                distribution_image = generate_display_frame(trial_directory, fake_examples_array,
                                                            unlabeled_predictions_array, test_predictions_array,
                                                            dnn_test_predictions_array, train_predictions_array,
                                                            dnn_train_predictions_array, step)
                gan_summary_writer.add_image('Distributions', distribution_image)

    print('Completed {}'.format(trial_directory))
    if settings.should_save_models:
        torch.save(DNN.state_dict(), os.path.join(trial_directory, 'DNN_model.pth'))
        torch.save(D.state_dict(), os.path.join(trial_directory, 'D_model.pth'))
        torch.save(G.state_dict(), os.path.join(trial_directory, 'G_model.pth'))


def clean_scientific_notation(string):
    regex = r'\.?0*e([+\-])0*([0-9])'
    string = re.sub(regex, r'e\g<1>\g<2>', string)
    string = re.sub(r'e\+', r'e', string)
    return string


for gradient_penalty_multiplier in [10]:
    for scale_multiplier in [1e-1]:
        scale_multiplier = scale_multiplier
        fake_multiplier = 1e0 * scale_multiplier
        unlabeled_multiplier = 1e0 * scale_multiplier
        settings_ = Settings()
        settings_.fake_loss_multiplier = fake_multiplier
        settings_.unlabeled_loss_multiplier = unlabeled_multiplier
        settings_.steps_to_run = 1000000
        settings_.learning_rate = 1e-5
        settings_.labeled_dataset_size = 15
        settings_.gradient_penalty_on = True
        settings_.gradient_penalty_multiplier = gradient_penalty_multiplier
        settings_.mean_offset = 0
        settings_.fake_loss_order = 1
        settings_.generator_training_step_period = 5
        settings_.trial_name = 'save ul {:e} fl {:e} {}le 1afgp{:e} zbrg{:e} lr {:e} seed 3'.format(unlabeled_multiplier, fake_multiplier, settings_.labeled_dataset_size, settings_.gradient_penalty_multiplier, settings_.mean_offset, settings_.learning_rate)
        settings_.trial_name = clean_scientific_notation(settings_.trial_name)
        try:
            run_srgan(settings_)
        except KeyboardInterrupt as error:
            print('\nGenerating video before quitting...', end='')
            # generate_video_from_frames(global_trial_directory)
            raise error
