"""
Code for preparing presentation stuff.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns

sns.set()
sns.set_style("dark")
dpi = 200

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

