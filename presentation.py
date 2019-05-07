"""
Code for preparing presentation stuff.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import norm
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


def srgan_loss_comparison():
    """Creates a plot which shows the theoretical loss curves for feature vector distances."""
    sns.set_style('darkgrid')
    figure, axes = plt.subplots(dpi=dpi)
    x_axis = np.arange(-10, 10, 0.01)

    l_g = -(np.log(np.abs(x_axis) + 1))
    l_unlabeled = np.power(x_axis, 2)

    axes.plot(x_axis, (l_g - l_g.min()) / (l_g.max() - l_g.min()), color=sns.color_palette()[4], label='$L_G$')
    axes.plot(x_axis, (l_unlabeled - l_unlabeled.min()) / (l_unlabeled.max() - l_unlabeled.min()),
              color=sns.color_palette()[5], label='$L_{fake}$')
    axes.legend().set_visible(True)
    axes.legend(loc='right')
    # axes.get_yaxis().set_ticks([])
    axes.set_ylabel('Loss')
    axes.set_xlabel('Feature difference')
    axes.get_yaxis().set_major_formatter(ticker.NullFormatter())
    axes.get_yaxis().grid(False)
    matplotlib2tikz.save(os.path.join('latex', 'srgan-losses.tex'))
    plt.show()
    plt.close(figure)


if __name__ == '__main__':
    plt.switch_backend('module://backend_interagg')
    srgan_loss_comparison()
