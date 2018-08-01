"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import select
import sys

import numpy as np
from scipy.stats import norm
from torch.nn import Module
from torch.optim import Adam, Optimizer
import torch
from torch.utils.data import Dataset, DataLoader

from settings import Settings
from utility import SummaryWriter, infinite_iter, gpu, make_directory_name_unique, MixtureModel, seed_all


class Experiment:
    """A class to manage an experimental trial."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.trial_directory: str = None
        self.dnn_summary_writer: SummaryWriter = None
        self.gan_summary_writer: SummaryWriter = None
        self.train_dataset: Dataset = None
        self.train_dataset_loader: DataLoader = None
        self.unlabeled_dataset: Dataset = None
        self.unlabeled_dataset_loader: DataLoader = None
        self.validation_dataset: Dataset = None
        self.DNN: Module = None
        self.DNN_optimizer: Optimizer = None
        self.D: Module = None
        self.D_optimizer: Optimizer = None
        self.G: Module = None
        self.G_optimizer: Optimizer = None
        self.signal_quit = False
        self.labeled_loss_function = None

        self.labeled_features = None
        self.unlabeled_features = None
        self.fake_features = None
        self.interpolates_features = None

    def train(self):
        """
        Run the SRGAN training for the experiment.
        """
        self.trial_directory = os.path.join(self.settings.logs_directory, self.settings.trial_name)
        if (self.settings.skip_completed_experiment and os.path.exists(self.trial_directory) and
                '/check' not in self.trial_directory):
            print('`{}` experiment already exists. Skipping...'.format(self.trial_directory))
            return
        self.trial_directory = make_directory_name_unique(self.trial_directory)
        print(self.trial_directory)
        os.makedirs(os.path.join(self.trial_directory, self.settings.temporary_directory))
        self.dnn_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'DNN'))
        self.gan_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'GAN'))
        self.dnn_summary_writer.summary_period = self.settings.summary_step_period
        self.gan_summary_writer.summary_period = self.settings.summary_step_period
        seed_all(0)

        dataset_setup = self.settings.application.dataset_setup
        model_setup = self.settings.application.model_setup
        validation_summaries = self.settings.application.validation_summaries
        self.labeled_loss_function = self.settings.application.labeled_loss_function

        (self.train_dataset, self.train_dataset_loader, self.unlabeled_dataset, self.unlabeled_dataset_loader,
            self.validation_dataset) = dataset_setup(self)
        DNN_model, D_model, G_model = model_setup()

        if self.settings.load_model_path:
            if not torch.cuda.is_available():
                map_location = 'cpu'
            else:
                map_location = None
            DNN_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'DNN_model.pth'),
                                                 map_location))
            D_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'D_model.pth'),
                                               map_location))
            G_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'G_model.pth'),
                                               map_location))
        self.G = G_model.to(gpu)
        self.D = D_model.to(gpu)
        self.DNN = DNN_model.to(gpu)
        d_lr = self.settings.learning_rate
        g_lr = d_lr

        # betas = (0.9, 0.999)
        weight_decay = 1e-2
        self.D_optimizer = Adam(self.D.parameters(), lr=d_lr, weight_decay=weight_decay)
        self.G_optimizer = Adam(self.G.parameters(), lr=g_lr)
        self.DNN_optimizer = Adam(self.DNN.parameters(), lr=d_lr, weight_decay=weight_decay)

        step_time_start = datetime.datetime.now()
        train_dataset_generator = infinite_iter(self.train_dataset_loader)
        unlabeled_dataset_generator = infinite_iter(self.unlabeled_dataset_loader)

        for step in range(self.settings.steps_to_run):
            # DNN.
            labeled_examples, labels = next(train_dataset_generator)
            labeled_examples, labels = labeled_examples.to(gpu), labels.to(gpu)
            self.dnn_training_step(labeled_examples, labels, step)
            # GAN.
            unlabeled_examples, _ = next(unlabeled_dataset_generator)
            unlabeled_examples = unlabeled_examples.to(gpu)
            self.gan_training_step(labeled_examples, labels, unlabeled_examples, step)

            if self.gan_summary_writer.is_summary_step():
                print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                step_time_start = datetime.datetime.now()

                self.D.eval()
                self.DNN.eval()
                self.G.eval()
                validation_summaries(self, step)
                self.D.train()
                self.DNN.train()
                self.G.train()
                while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if 'save' in line:
                        torch.save(self.DNN.state_dict(), os.path.join(self.trial_directory,
                                                                       'DNN_model_{}.pth'.format(step)))
                        torch.save(self.D.state_dict(), os.path.join(self.trial_directory,
                                                                     'D_model_{}.pth'.format(step)))
                        torch.save(self.G.state_dict(), os.path.join(self.trial_directory,
                                                                     'G_model_{}.pth'.format(step)))
                        print('\rSaved model for step {}...'.format(step))
                    if 'quit' in line:
                        self.signal_quit = True
                        print('\rQuit requested after current experiment...')

        print('Completed {}'.format(self.trial_directory))
        if self.settings.should_save_models:
            torch.save(self.DNN.state_dict(), os.path.join(self.trial_directory, 'DNN_model.pth'))
            torch.save(self.D.state_dict(), os.path.join(self.trial_directory, 'D_model.pth'))
            torch.save(self.G.state_dict(), os.path.join(self.trial_directory, 'G_model.pth'))

    def dnn_training_step(self, examples, labels, step):
        """Runs an individual round of DNN training."""
        self.dnn_summary_writer.step = step
        self.DNN_optimizer.zero_grad()
        dnn_predicted_labels = self.DNN(examples)
        dnn_loss = self.labeled_loss_function(dnn_predicted_labels, labels) * self.settings.labeled_loss_multiplier
        dnn_features = self.DNN.features
        dnn_loss.backward()
        self.DNN_optimizer.step()
        # Summaries.
        if self.dnn_summary_writer.is_summary_step():
            self.dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.item())
            self.dnn_summary_writer.add_scalar('Feature Norm/Labeled', dnn_features.norm(dim=1).mean().item())

    def gan_training_step(self, labeled_examples, labels, unlabeled_examples, step):
        """Runs an individual round of GAN training."""
        # Labeled.
        self.gan_summary_writer.step = step
        self.D_optimizer.zero_grad()
        labeled_loss = self.labeled_loss_calculation(labeled_examples, labels)
        # Unlabeled.
        unlabeled_loss = self.unlabeled_loss_calculation(unlabeled_examples)
        # Fake.
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        fake_loss = self.fake_loss_calculation(fake_examples)
        # Gradient penalty.
        alpha = torch.rand(2, device=gpu)
        alpha = alpha / alpha.sum(0)
        interpolates = (alpha[0] * unlabeled_examples.detach().requires_grad_() +
                        alpha[1] * fake_examples.detach().requires_grad_())
        interpolates_loss = self.interpolate_loss_calculation(interpolates)
        gradients = torch.autograd.grad(outputs=interpolates_loss, inputs=interpolates,
                                        grad_outputs=torch.ones_like(interpolates_loss, device=gpu),
                                        create_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.view(unlabeled_examples.size(0), -1).norm(dim=1) - 1) ** 2
                            ).mean() * self.settings.gradient_penalty_multiplier
        # Discriminator update.
        loss = labeled_loss + unlabeled_loss + fake_loss + gradient_penalty
        loss.backward()
        self.D_optimizer.step()
        # Generator.
        if step % self.settings.generator_training_step_period == 0:
            self.G_optimizer.zero_grad()
            z = torch.randn(unlabeled_examples.size(0), self.G.input_size).to(gpu)
            fake_examples = self.G(z)
            generator_loss = self.generator_loss_calculation(fake_examples, unlabeled_examples)
            generator_loss.backward()
            self.G_optimizer.step()
            if self.gan_summary_writer.is_summary_step():
                self.gan_summary_writer.add_scalar('Generator/Loss', generator_loss.item())
        # Summaries.
        if self.gan_summary_writer.is_summary_step():
            self.gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.item())
            self.gan_summary_writer.add_scalar('Feature Norm/Labeled',
                                               self.labeled_features.mean(0).norm().item())
            self.gan_summary_writer.add_scalar('Feature Norm/Unlabeled',
                                               self.unlabeled_features.mean(0).norm().item())
            self.gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.item())

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_labels = self.D(labeled_examples)
        self.labeled_features = self.D.features
        labeled_loss = self.labeled_loss_function(predicted_labels, labels) * self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, unlabeled_examples):
        """Calculates the unlabeled loss."""
        _ = self.D(unlabeled_examples)
        self.unlabeled_features = self.D.features
        unlabeled_loss = feature_distance_loss(self.unlabeled_features, self.labeled_features,
                                               order=self.settings.unlabeled_loss_order
                                               ) * self.settings.unlabeled_loss_multiplier
        return unlabeled_loss

    def fake_loss_calculation(self, fake_examples):
        """Calculates the fake loss."""
        _ = self.D(fake_examples.detach())
        self.fake_features = self.D.features
        fake_loss = feature_distance_loss(self.unlabeled_features, self.fake_features,
                                          scale=self.settings.normalize_fake_loss, order=self.settings.fake_loss_order
                                          ).neg() * self.settings.fake_loss_multiplier
        return fake_loss

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        _ = self.D(interpolates)
        self.interpolates_features = self.D.features
        interpolates_loss = feature_distance_loss(self.unlabeled_features, self.interpolates_features,
                                                  scale=self.settings.normalize_fake_loss,
                                                  order=self.settings.fake_loss_order
                                                  ).neg() * self.settings.fake_loss_multiplier
        return interpolates_loss

    def generator_loss_calculation(self, fake_examples, unlabeled_examples):
        """Calculates the generator's loss."""
        _ = self.D(fake_examples)
        self.fake_features = self.D.features
        _ = self.D(unlabeled_examples)
        detached_unlabeled_features = self.D.features.detach()
        generator_loss = feature_distance_loss(detached_unlabeled_features, self.fake_features,
                                               order=self.settings.generator_loss_order)
        return generator_loss


def unit_vector(vector):
    """Gets the unit vector version of a vector."""
    return vector.div(vector.norm() + 1e-10)


def angle_between(vector0, vector1):
    """Calculates the angle between two vectors."""
    unit_vector0 = unit_vector(vector0)
    unit_vector1 = unit_vector(vector1)
    epsilon = 1e-6
    return unit_vector0.dot(unit_vector1).clamp(-1.0 + epsilon, 1.0 - epsilon).acos()


def feature_distance_loss(base_features, other_features, order=2, base_noise=0, scale=False):
    """Calculate the loss based on the distance between feature vectors."""
    base_mean_features = base_features.mean(0)
    other_mean_features = other_features.mean(0)
    if base_noise:
        base_mean_features += torch.normal(torch.zeros_like(base_mean_features), base_mean_features * base_noise)
    mean_feature_distance = (base_mean_features - other_mean_features).abs().pow(2).sum().pow(1 / 2)
    if scale:
        scale_epsilon = 1e-10
        mean_feature_distance /= (base_mean_features.norm() + other_mean_features.norm() + scale_epsilon)
    if order < 1:
        order_epsilon = 1e-2
        mean_feature_distance += order_epsilon
    return mean_feature_distance.pow(order)


def feature_angle_loss(base_features, other_features, target=0, summary_writer=None):
    """Calculate the loss based on the angle between feature vectors."""
    angle = angle_between(base_features.mean(0), other_features.mean(0))
    if summary_writer:
        summary_writer.add_scalar('Feature Vector/Angle', angle.item())
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
