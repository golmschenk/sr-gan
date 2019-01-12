"""
Code for preparing presentation stuff.
"""
import os
import re
import imageio
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import seaborn as sns
import matplotlib2tikz

from utility import MixtureModel

plt.switch_backend('Agg')
sns.set()
dpi = 300


def generate_polynomial_concept_images():
    """Generates examples of polynomials and how they look for use in presentations about the method."""
    np.random.seed(0)
    sns.set_style('darkgrid')
    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(-1, 1, 0.001)
    axes.plot(x_axis, (-1 * (x_axis ** 4)) + (-1 * (x_axis ** 3)) + (2 * (x_axis ** 2)) + x_axis,
              color=sns.color_palette()[4])
    observation_color = sns.xkcd_rgb['medium grey']
    for observation in np.linspace(-1, 1, num=10):
        axes.plot([observation, observation],
                  [0, -1 * observation ** 4 + -1 * observation ** 3 + 2 * observation ** 2 + observation],
                  color=observation_color)
    plt.savefig('polynomial_with_observations.png', dpi=dpi)
    matplotlib2tikz.save(os.path.join('latex', 'polynomial_example.tex'))
    plt.close(figure)


def generate_single_peak_double_peak(mean_offset=3):
    """Creates a display of a single peak normal distribution surrounded by a double peak one."""
    sns.set_style('darkgrid')
    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(-5, 5, 0.001)
    axes.plot(x_axis, norm(0, 1).pdf(x_axis), color=sns.color_palette()[0])
    axes.plot(x_axis, MixtureModel([norm(-mean_offset, 1), norm(mean_offset, 1)]).pdf(x_axis),
              color=sns.color_palette()[1], label='HHZ 1')
    matplotlib2tikz.save(os.path.join('latex', 'single_peak_double_peak.tex'))
    plt.show()
    plt.close(figure)


def generate_display_frame(fake_examples, unlabeled_predictions, test_predictions, dnn_test_predictions,
                           train_predictions, dnn_train_predictions, step):
    """Generates an image of the distribution predictions during training."""
    sns.set_style('darkgrid')
    bandwidth = 0.1
    fake_a3 = np.transpose(np.polyfit(np.linspace(-1, 1, num=10), np.transpose(fake_examples[:, :10]), 3))
    x_axis_limits = [-4, 4]
    x_axis = np.arange(*x_axis_limits, 0.001)
    figure, axes = plt.subplots(dpi=dpi)
    axes.text(0.98, 0.98, 'Step: {}'.format(step), horizontalalignment='right', verticalalignment='top',
              family='monospace', fontsize=10, transform=axes.transAxes)
    axes.plot(x_axis, MixtureModel([uniform(-2, 1), uniform(1, 1)]).pdf(x_axis), color=sns.color_palette()[0],
              label='Real Data Distribution')
    try:
        axes = sns.kdeplot(fake_a3[0, :], ax=axes, color=sns.color_palette()[4], bw=bandwidth,
                           label='Fake Data Distribution')
    except ValueError:
        pass
    axes = sns.kdeplot(unlabeled_predictions, ax=axes, color=sns.color_palette()[1], bw=bandwidth,
                       label='Unlabeled Predictions')
    axes = sns.kdeplot(test_predictions, ax=axes, color=sns.color_palette()[2], bw=bandwidth,
                       label='GAN Test Predictions')
    axes = sns.kdeplot(train_predictions, ax=axes, color=sns.color_palette()[2], linewidth=0.5, bw=bandwidth,
                       label='GAN Train Predictions')
    axes = sns.kdeplot(dnn_test_predictions, ax=axes, color=sns.color_palette()[3], bw=bandwidth,
                       label='DNN Test Predictions')
    axes = sns.kdeplot(dnn_train_predictions, ax=axes, color=sns.color_palette()[3], linewidth=0.5, bw=bandwidth,
                       label='DNN Train Predictions')
    axes.set_xlim(*x_axis_limits)
    axes.set_ylim(0, 1)
    axes.legend(loc='upper left')
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    image_array = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image_array = image_array.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    plt.close(figure)
    return image_array


def natural_sort_key(string, natural_sort_regex=re.compile('([0-9]+)')):
    """A key for sorting string numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(natural_sort_regex, string)]


def generate_video_from_frames(trial_directory, settings):
    """Generates a video from the presentation frames."""
    fps = 20
    file_names = [file for file in os.listdir(os.path.join(trial_directory, settings.temporary_directory))
                  if file.endswith('.png')]
    file_names.sort(key=natural_sort_key)
    video_writer = imageio.get_writer(os.path.join(trial_directory, '{}.mp4'.format(os.path.basename(trial_directory))),
                                      fps=fps)
    for file_name in file_names:
        image = imageio.imread(os.path.join(trial_directory, '{}/{}'.format(settings.temporary_directory, file_name)))
        video_writer.append_data(image)
    video_writer.close()
    shutil.rmtree(os.path.join(trial_directory, settings.temporary_directory))
    print('\nMeans Video Complete.')


def nash_equilibrium_plot():
    """Creates a plot of Nash equilibrium problem of a GAN for presentation purposes."""
    figure, axes = plt.subplots(dpi=dpi)
    points = np.array([(1, 0), (0, -1), (-1, 0), (0, 1), (1, 0)])
    axes.plot(*zip(*points), color=sns.color_palette()[0], label='DNN')
    matplotlib2tikz.save(os.path.join('latex', 'nash_equilibrium.tex'))
    plt.show()
    plt.close(figure)


def map_comparisons(sigmas=None):
    """Creates a plot comparing various choices of value maps for the crowd analysis case."""
    if sigmas is None:
        sigmas = [0.4, 1, 2, 4, 6, 8]
    sns.set_style('darkgrid')
    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(0, 4, 0.001)
    normals = []
    for sigma in sigmas:
        normal_ = norm(0, sigma)
        normals.append(normal_)
    mixture = MixtureModel(normals)
    axes.plot(x_axis, mixture.pdf(x_axis) / mixture.pdf(x_axis).max(), color=sns.color_palette()[0])
    axes.plot(x_axis, normals[0].pdf(x_axis) / (normals[0].pdf(x_axis).max()), color=sns.color_palette()[1])

    axes.plot(x_axis, normals[-1].pdf(x_axis) / (normals[-1].pdf(x_axis).max()), color=sns.color_palette()[1])
    axes.plot(x_axis, (1 / (x_axis + 1)) / (1 / (x_axis + 1)).max(), color=sns.color_palette()[2])
    axes.set_ylabel('Map value')
    axes.set_xlabel('Distance from head position')
    matplotlib2tikz.save(os.path.join('latex', 'mapcomparisons.tex'))
    plt.show()
    plt.close(figure)


if __name__ == '__main__':
    plt.switch_backend('module://backend_interagg')
    map_comparisons()
