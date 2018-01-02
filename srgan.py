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
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter as SummaryWriter_
import torch

from settings import Settings
from data import ToyDataset, MixtureModel
from hardware import gpu, cpu
from presentation import generate_video_from_frames, generate_display_frame

global_trial_directory = None
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


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

    def add_image(self, tag, img_tensor, global_step=None):
        if global_step is None:
            global_step = self.step
        if self.step % self.summary_period == 0:
            super().add_image(tag, img_tensor, global_step)

def run_rsgan(settings):
    """
    :param settings: The settings object.
    :type settings: Settings
    """
    datetime_string = datetime.datetime.now().strftime('y%Ym%md%dh%Hm%Ms%S')
    trial_directory = os.path.join(settings.logs_directory, '{} {}'.format(settings.trial_name, datetime_string))
    os.makedirs(os.path.join(trial_directory, 'presentation'))
    global global_trial_directory
    global_trial_directory = trial_directory
    os.makedirs(os.path.join(trial_directory, settings.temporary_directory))
    dnn_summary_writer = SummaryWriter(os.path.join(trial_directory, 'DNN'))
    gan_summary_writer = SummaryWriter(os.path.join(trial_directory, 'GAN'))
    dnn_summary_writer.summary_period = settings.summary_step_period
    gan_summary_writer.summary_period = settings.summary_step_period
    observation_count = 10
    noise_size = 10

    train_dataset = ToyDataset(dataset_size=settings.labeled_dataset_size, observation_count=observation_count)
    train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)

    unlabeled_dataset = ToyDataset(dataset_size=settings.unlabeled_dataset_size, observation_count=observation_count)
    unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True)

    test_dataset = ToyDataset(settings.test_dataset_size, observation_count)

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
            fake_examples = gpu(Variable(torch.from_numpy(fake_examples)))
            return fake_examples

    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(observation_count, 32)
            self.linear3 = Linear(32, 8)
            self.linear4 = Linear(8, 2)
            self.feature_layer = None
            self.gradient_sum = gpu(Variable(torch.zeros(1)))
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
            self.gradient_sum = gpu(Variable(torch.zeros(1)))

    class WeightClipper(object):
        def __call__(self, module):
            if hasattr(module, 'weight'):
                w = module.weight.data
                w.clamp_(min=-1, max=1)

    clipper = WeightClipper()

    G = gpu(Generator())
    D = gpu(MLP())
    DNN = gpu(MLP())
    d_lr = 1e-4
    g_lr = d_lr

    betas = (0.9, 0.999)
    weight_decay = 0
    D_optimizer = Adam(D.parameters(), lr=d_lr, betas=betas, weight_decay=weight_decay)
    G_optimizer = Adam(G.parameters(), lr=g_lr, betas=betas)
    DNN_optimizer = Adam(DNN.parameters(), lr=d_lr, betas=betas, weight_decay=weight_decay)

    # learning_rate_multiplier_function = lambda epoch: 0.1 ** (epoch / 1000000)
    # dnn_scheduler = lr_scheduler.LambdaLR(DNN_optimizer, lr_lambda=learning_rate_multiplier_function)
    # dnn_scheduler.step(0)
    # d_scheduler = lr_scheduler.LambdaLR(D_optimizer, lr_lambda=learning_rate_multiplier_function)
    # d_scheduler.step(0)
    # g_scheduler = lr_scheduler.LambdaLR(G_optimizer, lr_lambda=learning_rate_multiplier_function)
    # g_scheduler.step(0)

    step_time_start = datetime.datetime.now()
    print(trial_directory)

    for step in range(settings.steps_to_run):
        labeled_examples, labels = next(iter(train_dataset_loader))
        # DNN.
        gan_summary_writer.step = step
        dnn_summary_writer.step = step
        if step % settings.summary_step_period == 0 and step != 0:
            print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
            step_time_start = datetime.datetime.now()
        DNN_optimizer.zero_grad()
        dnn_predicted_labels = DNN(gpu(Variable(labeled_examples)))
        dnn_loss = (dnn_predicted_labels[:, 0] - gpu(Variable(labels[:, 0]))).abs().mean()
        dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.data[0])
        dnn_loss.backward()
        DNN_optimizer.step()
        # Labeled.
        D_optimizer.zero_grad()
        predicted_labels = D(gpu(Variable(labeled_examples)))
        labeled_loss = (predicted_labels[:, 0] - gpu(Variable(labels[:, 0]))).abs().mean()
        gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.data[0])
        D.zero_gradient_sum()
        labeled_loss.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Labeled', D.gradient_sum.data[0])
        # Unlabeled.
        _ = D(gpu(Variable(labeled_examples)))
        labeled_feature_layer = D.feature_layer
        unlabeled_examples, _ = next(iter(unlabeled_dataset_loader))
        _ = D(gpu(Variable(unlabeled_examples)))
        unlabeled_feature_layer = D.feature_layer
        unlabeled_loss = (unlabeled_feature_layer.mean(0) - labeled_feature_layer.mean(0)).pow(2).sum().pow(1/2)
        gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.data[0])
        D.zero_gradient_sum()
        unlabeled_loss.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Unlabeled', D.gradient_sum.data[0])
        # Fake.
        _ = D(gpu(Variable(unlabeled_examples)))
        unlabeled_feature_layer = D.feature_layer
        z = torch.randn(settings.batch_size, noise_size)
        fake_examples = G(gpu(Variable(z)))
        _ = D(fake_examples.detach())
        fake_feature_layer = D.feature_layer
        fake_loss = (unlabeled_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).sum().pow(1/2).neg()
        gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.data[0])
        D.zero_gradient_sum()
        fake_loss.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Fake', D.gradient_sum.data[0])
        # Gradient penalty.
        alpha = gpu(Variable(torch.rand(2, settings.batch_size, 1)))
        alpha = alpha / alpha.sum(0)
        interpolates = (alpha[0] * gpu(Variable(unlabeled_examples, requires_grad=True)) +
                        alpha[1] * gpu(Variable(fake_examples.detach().data, requires_grad=True)))
        interpolates_predictions = D(interpolates)
        gradients = torch.autograd.grad(outputs=interpolates_predictions, inputs=interpolates,
                                        grad_outputs=gpu(torch.ones(interpolates_predictions.size())),
                                        create_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1e1
        D.zero_gradient_sum()
        gradient_penalty.backward()
        gan_summary_writer.add_scalar('Gradient Sums/Gradient Penalty', D.gradient_sum.data[0])
        # Discriminator update.
        D_optimizer.step()
        # D.apply(clipper)
        # Generator.
        if step % 5 == 0:
            G_optimizer.zero_grad()
            _ = D(gpu(Variable(unlabeled_examples)))
            unlabeled_feature_layer = D.feature_layer.detach()
            z = torch.randn(settings.batch_size, noise_size)
            fake_examples = G(gpu(Variable(z)))
            _ = D(fake_examples)
            fake_feature_layer = D.feature_layer
            generator_loss = (unlabeled_feature_layer.mean(0) - fake_feature_layer.mean(0)).pow(2).sum().pow(1/2)
            gan_summary_writer.add_scalar('Generator/Loss', generator_loss.data[0])
            generator_loss.backward()
            G_optimizer.step()

        if dnn_summary_writer.step % dnn_summary_writer.summary_period == 0 or dnn_summary_writer.step % settings.presentation_step_period == 0:
            dnn_predicted_train_labels = cpu(DNN(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
            dnn_train_label_errors = np.mean(np.abs(dnn_predicted_train_labels - train_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Train Error/Mean', dnn_train_label_errors.data[0])
            dnn_summary_writer.add_scalar('Train Error/Std', dnn_train_label_errors.data[1])
            dnn_predicted_test_labels = cpu(DNN(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
            dnn_test_label_errors = np.mean(np.abs(dnn_predicted_test_labels - test_dataset.labels), axis=0)
            dnn_summary_writer.add_scalar('Test Error/Mean', dnn_test_label_errors.data[0])
            dnn_summary_writer.add_scalar('Test Error/Std', dnn_test_label_errors.data[1])

            predicted_train_labels = cpu(D(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
            gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Train Error/Mean', gan_train_label_errors.data[0])
            gan_summary_writer.add_scalar('Train Error/Std', gan_train_label_errors.data[1])
            predicted_test_labels = cpu(D(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
            gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)
            gan_summary_writer.add_scalar('Test Error/Mean', gan_test_label_errors.data[0])
            gan_summary_writer.add_scalar('Test Error/Std', gan_test_label_errors.data[1])
            gan_summary_writer.add_scalar('Test Error/Ratio Mean GAN DNN', gan_test_label_errors.data[0] / dnn_test_label_errors.data[0])

            if dnn_summary_writer.step % settings.presentation_step_period == 0:
                z = torch.randn(settings.test_dataset_size, noise_size)
                fake_examples = G(gpu(Variable(z)))
                fake_examples_array = cpu(fake_examples.data).numpy()
                unlabeled_examples_array = unlabeled_dataset.examples[:settings.test_dataset_size]
                unlabeled_examples = torch.from_numpy(unlabeled_examples_array.astype(np.float32))
                unlabeled_predictions = D(gpu(Variable(unlabeled_examples)))
                unlabeled_predictions_array = cpu(unlabeled_predictions.data).numpy()
                test_predictions_array = predicted_test_labels
                train_predictions_array = predicted_train_labels
                dnn_test_predictions_array = dnn_predicted_test_labels
                dnn_train_predictions_array = dnn_predicted_train_labels
                distribution_image = generate_display_frame(trial_directory, fake_examples_array, unlabeled_predictions_array, test_predictions_array, dnn_test_predictions_array, train_predictions_array, dnn_train_predictions_array, step)
                gan_summary_writer.add_image('Distributions', distribution_image)

        # dnn_scheduler.step(step)
        # d_scheduler.step(step)
        # g_scheduler.step(step)

    predicted_train_labels = cpu(DNN(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
    dnn_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
    predicted_test_labels = cpu(DNN(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
    dnn_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)

    predicted_train_labels = cpu(D(gpu(Variable(torch.from_numpy(train_dataset.examples.astype(np.float32))))).data).numpy()
    gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels), axis=0)
    predicted_test_labels = cpu(D(gpu(Variable(torch.from_numpy(test_dataset.examples.astype(np.float32))))).data).numpy()
    gan_test_label_errors = np.mean(np.abs(predicted_test_labels - test_dataset.labels), axis=0)

    print('DNN Train: {}'.format(np.mean(dnn_train_label_errors, axis=0)))
    print('DNN Test: {}'.format(np.mean(dnn_test_label_errors, axis=0)))
    print('GAN Train: {}'.format(np.mean(gan_train_label_errors, axis=0)))
    print('GAN Test: {}'.format(np.mean(gan_test_label_errors, axis=0)))

    generate_video_from_frames(global_trial_directory)
    print('Completed {}'.format(trial_directory))

settings = Settings()
try:
    run_rsgan(settings)
except KeyboardInterrupt:
    print('Generating video before quitting...')
    generate_video_from_frames(global_trial_directory)
    exit()
