"""
Code for preparing presentation stuff.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import seaborn as sns

sns.set()
sns.set_style("dark")

x_axis = np.arange(-3, 3, 0.001)
plt.plot([-0.5, -0.5], [0, norm.pdf(-0.5, 0, 1)], color=sns.color_palette()[1])
plt.plot([0.5, 0.5], [0, norm.pdf(0.5, 0, 1)], color=sns.color_palette()[2])
plt.plot([-0.2, -0.2], [0, norm.pdf(-0.2, 0, 1)], color=sns.color_palette()[4])
plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color=sns.color_palette()[0])
plt.show()

x_axis = np.arange(0, 6, 0.001)
plt.plot([1, 1], [0, gamma.pdf(1, 2)], color=sns.color_palette()[1])
plt.plot([0.7, 0.7], [0, gamma.pdf(0.7, 2)], color=sns.color_palette()[2])
plt.plot([2, 2], [0, gamma.pdf(2, 2)], color=sns.color_palette()[4])
plt.plot(x_axis, gamma.pdf(x_axis, 2), color=sns.color_palette()[0])
plt.show()

x_axis = np.arange(-4.5, 4.5, 0.001)
plt.plot(x_axis, norm.pdf(x_axis, -0.5, 1), color=sns.color_palette()[1])
plt.plot(x_axis, norm.pdf(x_axis, 0.5, 0.7), color=sns.color_palette()[2])
plt.plot(x_axis, norm.pdf(x_axis, -0.2, 2), color=sns.color_palette()[4])
plt.show()

