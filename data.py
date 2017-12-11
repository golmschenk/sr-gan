"""
Code for the data generating models.
"""
import numpy as np


def generate_simple_data(number_of_examples, number_of_observations):
    means = np.random.normal(size=[number_of_examples, 1])
    stds = np.random.gamma(shape=2, size=[number_of_examples, 1])
    examples = np.random.normal(means, stds, size=[number_of_examples, number_of_observations])
    examples.sort(axis=1)
    labels = np.concatenate((means, stds), axis=1)
    return examples, labels