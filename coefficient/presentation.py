"""Code for summary charts for the coefficient application."""
import os
import re
import shutil
import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import uniform

from utility import MixtureModel


plt.switch_backend('Agg')
sns.set()
dpi = 300


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
