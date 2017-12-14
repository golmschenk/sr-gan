"""
Code for preparing presentation stuff.
"""
import os
import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns

import settings
from data import MixtureModel

sns.set()
dpi = 200


def generate_data_concept_images():
    sns.set_style('dark')
    plt.figure()
    x_axis = np.arange(-3, 3, 0.001)
    plt.plot([-0.5, -0.5], [0, norm.pdf(-0.5, 0, 1)], color=sns.color_palette()[1])
    plt.plot([0.5, 0.5], [0, norm.pdf(0.5, 0, 1)], color=sns.color_palette()[2])
    plt.plot([-0.2, -0.2], [0, norm.pdf(-0.2, 0, 1)], color=sns.color_palette()[4])
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color=sns.color_palette()[0])
    plt.savefig('mean_generating_distribution.png', dpi=dpi)

    plt.figure()
    x_axis = np.arange(0, 6, 0.001)
    plt.plot([1, 1], [0, gamma.pdf(1, 2)], color=sns.color_palette()[1])
    plt.plot([0.7, 0.7], [0, gamma.pdf(0.7, 2)], color=sns.color_palette()[2])
    plt.plot([2, 2], [0, gamma.pdf(2, 2)], color=sns.color_palette()[4])
    plt.plot(x_axis, gamma.pdf(x_axis, 2), color=sns.color_palette()[0])
    plt.savefig('std_generating_distribution.png', dpi=dpi)

    plt.figure()
    x_axis = np.arange(-4.5, 4.5, 0.001)
    plt.plot(x_axis, norm.pdf(x_axis, -0.5, 1), color=sns.color_palette()[1])
    plt.plot(x_axis, norm.pdf(x_axis, 0.5, 0.7), color=sns.color_palette()[2])
    plt.plot(x_axis, norm.pdf(x_axis, -0.2, 2), color=sns.color_palette()[4])
    plt.savefig('example_normals.png', dpi=dpi)

    plt.figure()
    x_axis = np.arange(-4, 3, 0.001)
    observations = [-0.97100138, -1.20760565, -1.67125, -0.35949918, 1.04644455,
                    -0.06357208, -1.33066351, -1.06934841, -2.8277416, -0.67354897]
    observation_color = sns.xkcd_rgb['medium grey']
    for observation in observations:
        plt.plot([observation, observation], [0, norm.pdf(observation, -0.5, 1)], color=observation_color)
    plt.plot(x_axis, norm.pdf(x_axis, -0.5, 1), color=sns.color_palette()[1])
    plt.savefig('normal_with_observations.png', dpi=dpi)


def generate_learning_process_images():
    sns.set_style('darkgrid')
    stride = 1
    fps = 20
    bandwidth = 0.1
    fake_examples = np.load(os.path.join(settings.temporary_directory, 'fake_examples.npy'), mmap_mode='r')[::stride]
    unlabeled_predictions = np.load(os.path.join(settings.temporary_directory, 'unlabeled_predictions.npy'), mmap_mode='r')[::stride]
    test_predictions = np.load(os.path.join(settings.temporary_directory, 'test_predictions.npy'), mmap_mode='r')[::stride]
    dnn_test_predictions = np.load(os.path.join(settings.temporary_directory, 'dnn_test_predictions.npy'), mmap_mode='r')[::stride]
    train_predictions = np.load(os.path.join(settings.temporary_directory, 'train_predictions.npy'), mmap_mode='r')[::stride]
    dnn_train_predictions = np.load(os.path.join(settings.temporary_directory, 'dnn_train_predictions.npy'), mmap_mode='r')[::stride]
    fake_means = fake_examples.mean(axis=2)
    fake_stds = fake_examples.std(axis=2)

    x_axis_limits = [-6, 6]
    x_axis = np.arange(*x_axis_limits, 0.001)
    for step_index in range(fake_examples.shape[0]):
        print('\rGenerating image {} of {}...'.format(step_index, fake_examples.shape[0]), end='')
        figure, axes = plt.subplots()
        axes.plot(x_axis, MixtureModel([norm(-3, 1), norm(3, 1)]).pdf(x_axis), color=sns.color_palette()[0])
        axes = sns.kdeplot(fake_means[step_index], ax=axes, color=sns.color_palette()[4], bw=bandwidth)
        axes = sns.kdeplot(unlabeled_predictions[step_index, :, 0], ax=axes, color=sns.color_palette()[1], bw=bandwidth)
        axes = sns.kdeplot(test_predictions[step_index, :, 0], ax=axes, color=sns.color_palette()[2], bw=bandwidth)
        axes = sns.kdeplot(dnn_test_predictions[step_index, :, 0], ax=axes, color=sns.color_palette()[3], bw=bandwidth)
        axes = sns.kdeplot(train_predictions[step_index, :, 0], ax=axes, color=sns.color_palette()[2], linewidth=0.5, bw=bandwidth)
        axes = sns.kdeplot(dnn_train_predictions[step_index, :, 0], ax=axes, color=sns.color_palette()[3], linewidth=0.5, bw=bandwidth)
        axes.set_xlim(*x_axis_limits)
        axes.set_ylim(0, 0.5)
        plt.savefig('presentation/{}.png'.format(step_index), dpi=dpi, ax=axes)
        plt.close(figure)
    video_writer = imageio.get_writer('means.mp4', fps=fps)
    for image_index in range(fake_means.shape[0]):
        image = imageio.imread('presentation/{}.png'.format(image_index))
        video_writer.append_data(image)
    video_writer.close()
    print('\nMeans Video Complete.')

    x_axis_limits = [0, 7]
    x_axis = np.arange(*x_axis_limits, 0.001)
    for step_index in range(fake_examples.shape[0]):
        print('\rGenerating image {} of {}...'.format(step_index, fake_examples.shape[0]), end='')
        figure, axes = plt.subplots()
        axes.plot(x_axis, MixtureModel([gamma(2)]).pdf(x_axis), color=sns.color_palette()[0])
        axes = sns.kdeplot(fake_stds[step_index], ax=axes, color=sns.color_palette()[4], bw=bandwidth)
        axes = sns.kdeplot(unlabeled_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[1], bw=bandwidth)
        axes = sns.kdeplot(test_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[2], bw=bandwidth)
        axes = sns.kdeplot(dnn_test_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[3], bw=bandwidth)
        axes = sns.kdeplot(train_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[2], linewidth=0.5, bw=bandwidth)
        axes = sns.kdeplot(dnn_train_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[3], linewidth=0.5, bw=bandwidth)
        axes.set_xlim(*x_axis_limits)
        axes.set_ylim(0, 0.5)
        plt.savefig('presentation/{}.png'.format(step_index), dpi=dpi, ax=axes)
        plt.close(figure)
    video_writer = imageio.get_writer('stds.mp4', fps=fps)
    for image_index in range(fake_means.shape[0]):
        image = imageio.imread('presentation/{}.png'.format(image_index))
        video_writer.append_data(image)
    video_writer.close()
    print('\nStds Video Complete.')


if __name__ == '__main__':
    # generate_data_concept_images()
    generate_learning_process_images()