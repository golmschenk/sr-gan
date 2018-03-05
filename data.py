"""
Code for the data generating models.
"""
import random

import numpy as np
import time
import torch
from scipy.stats import rv_continuous, norm, gamma, uniform
from torch.utils.data import Dataset

from settings import Settings

settings = Settings()

irrelevant_data_multiplier = 5


class ToyDataset(Dataset):
    def __init__(self, dataset_size, observation_count, seed=None):
        seed_all(seed)
        self.examples, self.labels = generate_polynomial_examples(dataset_size, observation_count)
        if self.labels.shape[0] < settings.batch_size:
            repeats = settings.batch_size / self.labels.shape[0]
            self.examples = np.repeat(self.examples, repeats, axis=0)
            self.labels = np.repeat(self.labels, repeats, axis=0)
        self.length = self.labels.shape[0]

    def __getitem__(self, index):
        return self.examples[index], self.labels[index]

    def __len__(self):
        return self.length


def generate_simple_data(number_of_examples, number_of_observations):
    means = np.random.normal(size=[number_of_examples, 1])
    stds = np.random.gamma(shape=2, size=[number_of_examples, 1])
    examples = np.random.normal(means, stds, size=[number_of_examples, number_of_observations])
    labels = np.concatenate((means, stds), axis=1)
    return examples, labels


def generate_double_peak_data(number_of_examples, number_of_observations):
    double_peak_normal = MixtureModel([norm(-3, 1), norm(3, 1)])
    double_peak_gamma = MixtureModel([gamma(2), gamma(3, loc=4)])
    means = double_peak_normal.rvs(size=[number_of_examples, 1])
    stds = double_peak_gamma.rvs(size=[number_of_examples, 1])
    examples = np.random.normal(means, stds, size=[number_of_examples, number_of_observations])
    labels = np.concatenate((means, stds), axis=1)
    return examples, labels


def generate_double_mean_single_std_data_normals(number_of_examples, number_of_observations):
    mean_model = MixtureModel([norm(-3, 1), norm(3, 1)])
    std_model = MixtureModel([gamma(2)])
    means = mean_model.rvs(size=[number_of_examples, 1]).astype(dtype=np.float32)
    stds = std_model.rvs(size=[number_of_examples, 1]).astype(dtype=np.float32)
    examples = np.random.normal(means, stds, size=[number_of_examples, number_of_observations]).astype(dtype=np.float32)
    labels = np.concatenate((means, stds), axis=1)
    return examples, labels


def generate_double_mean_single_std_data(number_of_examples, number_of_observations):
    mean_model = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    std_model = MixtureModel([uniform(loc=1, scale=1)])
    middles = mean_model.rvs(size=[number_of_examples, 1]).astype(dtype=np.float32)
    widths = std_model.rvs(size=[number_of_examples, 1]).astype(dtype=np.float32)
    examples = np.random.uniform(middles - (widths / 2), middles + (widths / 2), size=[number_of_examples, number_of_observations]).astype(dtype=np.float32)
    labels = np.concatenate((middles, widths), axis=1)
    return examples, labels


def generate_polynomial_examples(number_of_examples, number_of_observations):
    a2, a3, a4 = generate_double_a2_single_a3_coefficients(number_of_examples)
    examples = generate_examples_from_coefficients(a2, a3, a4, number_of_observations)
    examples += np.random.normal(0, 0.1, examples.shape)
    return examples, a3[:, 0, :]


def generate_double_a2_single_a3_coefficients(number_of_examples):
    a2_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a3_distribution = MixtureModel([uniform(loc=-1, scale=2)])
    a2 = a2_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a3 = a3_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a4_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a4 = a4_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    return a2, a3, a4


def generate_examples_from_coefficients(a2, a3, a4, number_of_observations):
    x = np.linspace(-1, 1, num=number_of_observations)
    examples = x + (a2 * (x ** 2)) + (a3 * (x ** 3)) + (a4 * (x ** 4))
    examples = examples.reshape(examples.shape[0], number_of_observations * irrelevant_data_multiplier)
    return examples.astype(dtype=np.float32)


class MixtureModel(rv_continuous):
    def __init__(self, submodels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels

    def _pdf(self, x, **kwargs):
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def seed_all(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)