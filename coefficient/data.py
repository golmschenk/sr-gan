"""
Code for the polynomial coefficient data generating models.
"""

import numpy as np
from scipy.stats import uniform
from torch.utils.data import Dataset
from utility import MixtureModel, seed_all

irrelevant_data_multiplier = 5


class ToyDataset(Dataset):
    """The polynomial estimation dataset."""
    def __init__(self, dataset_size, observation_count, settings, seed=None):
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


def generate_polynomial_examples(number_of_examples, number_of_observations):
    """Generates polynomial estimation examples."""
    a2, a3, a4 = generate_double_a2_a3_a4_coefficients(number_of_examples)
    examples = generate_examples_from_coefficients(a2, a3, a4, number_of_observations)
    examples += np.random.normal(0, 0.1, examples.shape)
    labels = np.squeeze(a3[:, 0], axis=-1)
    return examples, labels


def generate_single_a3_double_a2_a4_coefficients(number_of_examples):
    """Generates coefficients with a single uniform distribution for a3 and double for a2 and a4."""
    a2_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a2 = a2_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a3_distribution = MixtureModel([uniform(loc=-1, scale=2)])
    a3 = a3_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a4_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a4 = a4_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    return a2, a3, a4


def generate_double_a2_a3_a4_coefficients(number_of_examples):
    """Generates coefficients with a double uniform distribution for a2, a3, and a4."""
    a2_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a2 = a2_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a3_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a3 = a3_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    a4_distribution = MixtureModel([uniform(-2, 1), uniform(1, 1)])
    a4 = a4_distribution.rvs(size=[number_of_examples, irrelevant_data_multiplier, 1]).astype(dtype=np.float32)
    return a2, a3, a4


def generate_examples_from_coefficients(a2, a3, a4, number_of_observations):
    """Generates polynomials from coefficients."""
    x = np.linspace(-1, 1, num=number_of_observations)
    examples = x + (a2 * (x ** 2)) + (a3 * (x ** 3)) + (a4 * (x ** 4))
    examples = examples.reshape(examples.shape[0], number_of_observations * irrelevant_data_multiplier)
    return examples.astype(dtype=np.float32)
