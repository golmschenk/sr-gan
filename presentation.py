"""
Code for preparing presentation stuff.
"""
import os
import imageio as imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns

from settings import Settings
from data import MixtureModel

sns.set()
dpi = 300
settings = Settings()


def generate_data_concept_images():
    np.random.seed(4)
    sns.set_style('dark')
    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(-6, 6, 0.001)
    mixture_model = MixtureModel([norm(-3, 1), norm(3, 1)])
    mean_samples = mixture_model.rvs(3)
    axes.plot([mean_samples[0], mean_samples[0]], [0, mixture_model.pdf(mean_samples[0], 0, 1)], color=sns.color_palette()[1])
    axes.plot([mean_samples[1], mean_samples[1]], [0, mixture_model.pdf(mean_samples[1], 0, 1)], color=sns.color_palette()[2])
    axes.plot([mean_samples[2], mean_samples[2]], [0, mixture_model.pdf(mean_samples[2], 0, 1)], color=sns.color_palette()[4])
    axes.plot(x_axis, mixture_model.pdf(x_axis), color=sns.color_palette()[0])
    plt.savefig('mean_generating_distribution.png', dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(0, 6, 0.001)
    gamma_distribution = gamma(2)
    std_samples = gamma_distribution.rvs(3)
    axes.plot([std_samples[0], std_samples[0]], [0, gamma_distribution.pdf(std_samples[0])], color=sns.color_palette()[1])
    axes.plot([std_samples[1], std_samples[1]], [0, gamma_distribution.pdf(std_samples[1])], color=sns.color_palette()[2])
    axes.plot([std_samples[2], std_samples[2]], [0, gamma_distribution.pdf(std_samples[2])], color=sns.color_palette()[4])
    axes.plot(x_axis, gamma.pdf(x_axis, 2), color=sns.color_palette()[0])
    plt.savefig('std_generating_distribution.png', dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(dpi=dpi)
    minimums = mean_samples - (3 * std_samples)
    maximums = mean_samples + (3 * std_samples)
    minimum = np.min(minimums)
    maximum = np.max(maximums)
    x_axis = np.arange(minimum, maximum, 0.001)
    example_norms = [norm(mean_samples[0], std_samples[0]), norm(mean_samples[1], std_samples[1]), norm(mean_samples[2], std_samples[2])]
    axes.plot(x_axis, example_norms[0].pdf(x_axis), color=sns.color_palette()[1])
    axes.plot(x_axis, example_norms[1].pdf(x_axis), color=sns.color_palette()[2])
    axes.plot(x_axis, example_norms[2].pdf(x_axis), color=sns.color_palette()[4])
    axes.set_ylim(0, 0.3)
    plt.savefig('example_normals.png', dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(mean_samples[0]-5, mean_samples[0]+5, 0.001)
    observations = example_norms[0].rvs(10)
    observation_color = sns.xkcd_rgb['medium grey']
    print(np.round(observations, 3))
    for observation in observations:
        axes.plot([observation, observation], [0, example_norms[0].pdf(observation)], color=observation_color)
    axes.plot(x_axis, example_norms[0].pdf(x_axis), color=sns.color_palette()[1])
    plt.savefig('normal_with_observations.png', dpi=dpi)
    plt.close(figure)


def generate_display_frame(trial_directory, fake_examples, unlabeled_predictions, test_predictions, dnn_test_predictions, train_predictions, dnn_train_predictions):
    step_index = len([file for file in os.listdir(os.path.join(trial_directory, 'presentation')) if file.endswith('.png')])
    sns.set_style('darkgrid')
    bandwidth = 0.1
    fake_means = fake_examples.mean(axis=1)
    x_axis_limits = [-6, 6]
    x_axis = np.arange(*x_axis_limits, 0.001)
    figure, axes = plt.subplots(dpi=dpi)
    axes.plot(x_axis, MixtureModel([norm(-3, 1), norm(3, 1)]).pdf(x_axis), color=sns.color_palette()[0])
    axes = sns.kdeplot(fake_means, ax=axes, color=sns.color_palette()[4], bw=bandwidth, label='Fake')
    axes = sns.kdeplot(unlabeled_predictions[:, 0], ax=axes, color=sns.color_palette()[1], bw=bandwidth, label='Unlabeled Predictions')
    axes = sns.kdeplot(test_predictions[:, 0], ax=axes, color=sns.color_palette()[2], bw=bandwidth, label='GAN Test Predictions')
    axes = sns.kdeplot(train_predictions[:, 0], ax=axes, color=sns.color_palette()[2], linewidth=0.5, bw=bandwidth, label='GAN Train Predictions')
    axes = sns.kdeplot(dnn_test_predictions[:, 0], ax=axes, color=sns.color_palette()[3], bw=bandwidth, label='DNN Test Predictions')
    axes = sns.kdeplot(dnn_train_predictions[:, 0], ax=axes, color=sns.color_palette()[3], linewidth=0.5, bw=bandwidth, label='DNN Train Predictions')
    axes.set_xlim(*x_axis_limits)
    axes.set_ylim(0, 0.5)
    axes.legend(loc='upper left')
    plt.savefig(os.path.join(trial_directory, 'presentation/{}.png'.format(step_index)), dpi=dpi, ax=axes)
    plt.close(figure)


def generate_learning_process_images(trial_directory):
    sns.set_style('darkgrid')
    stride = 1
    fps = 20
    bandwidth = 0.1
    fake_examples = np.load(os.path.join(trial_directory, settings.temporary_directory, 'fake_examples.npy'), mmap_mode='r')[::stride]
    unlabeled_predictions = np.load(os.path.join(trial_directory, settings.temporary_directory, 'unlabeled_predictions.npy'), mmap_mode='r')[::stride]
    test_predictions = np.load(os.path.join(trial_directory, settings.temporary_directory, 'test_predictions.npy'), mmap_mode='r')[::stride]
    dnn_test_predictions = np.load(os.path.join(trial_directory, settings.temporary_directory, 'dnn_test_predictions.npy'), mmap_mode='r')[::stride]
    train_predictions = np.load(os.path.join(trial_directory, settings.temporary_directory, 'train_predictions.npy'), mmap_mode='r')[::stride]
    dnn_train_predictions = np.load(os.path.join(trial_directory, settings.temporary_directory, 'dnn_train_predictions.npy'), mmap_mode='r')[::stride]
    fake_means = fake_examples.mean(axis=2)
    fake_stds = fake_examples.std(axis=2)
    os.makedirs(os.path.join(trial_directory, 'presentation'), exist_ok=True)

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
        plt.savefig(os.path.join(trial_directory, 'presentation/{}.png'.format(step_index)), dpi=dpi, ax=axes)
        plt.close(figure)
    video_writer = imageio.get_writer(os.path.join(trial_directory, 'means.mp4'), fps=fps)
    for image_index in range(fake_means.shape[0]):
        image = imageio.imread(os.path.join(trial_directory, 'presentation/{}.png'.format(image_index)))
        video_writer.append_data(image)
    video_writer.close()
    print('\nMeans Video Complete.')

    # x_axis_limits = [0, 7]
    # x_axis = np.arange(*x_axis_limits, 0.001)
    # for step_index in range(fake_examples.shape[0]):
    #     print('\rGenerating image {} of {}...'.format(step_index, fake_examples.shape[0]), end='')
    #     figure, axes = plt.subplots()
    #     axes.plot(x_axis, MixtureModel([gamma(2)]).pdf(x_axis), color=sns.color_palette()[0])
    #     axes = sns.kdeplot(fake_stds[step_index], ax=axes, color=sns.color_palette()[4], bw=bandwidth)
    #     axes = sns.kdeplot(unlabeled_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[1], bw=bandwidth)
    #     axes = sns.kdeplot(test_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[2], bw=bandwidth)
    #     axes = sns.kdeplot(dnn_test_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[3], bw=bandwidth)
    #     axes = sns.kdeplot(train_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[2], linewidth=0.5, bw=bandwidth)
    #     axes = sns.kdeplot(dnn_train_predictions[step_index, :, 1], ax=axes, color=sns.color_palette()[3], linewidth=0.5, bw=bandwidth)
    #     axes.set_xlim(*x_axis_limits)
    #     axes.set_ylim(0, 0.7)
    #     plt.savefig(os.path.join(trial_directory, 'presentation/{}.png'.format(step_index)), dpi=dpi, ax=axes)
    #     plt.close(figure)
    # video_writer = imageio.get_writer(os.path.join(trial_directory, 'stds.mp4'), fps=fps)
    # for image_index in range(fake_means.shape[0]):
    #     image = imageio.imread(os.path.join(trial_directory, 'presentation/{}.png'.format(image_index)))
    #     video_writer.append_data(image)
    # video_writer.close()
    # print('\nStds Video Complete.')

def generate_video_from_frames(trial_directory):
    fps = 20
    number_of_frames = len([file for file in os.listdir(os.path.join(trial_directory, 'presentation')) if file.endswith('.png')])
    video_writer = imageio.get_writer(os.path.join(trial_directory, 'means.mp4'), fps=fps)
    for image_index in range(number_of_frames):
        image = imageio.imread(os.path.join(trial_directory, 'presentation/{}.png'.format(image_index)))
        video_writer.append_data(image)
    video_writer.close()
    print('\nMeans Video Complete.')


if __name__ == '__main__':
    generate_data_concept_images()
