import numpy as np
from scipy.optimize import curve_fit

from coefficient.coefficient_data import ToyDataset



for test_dataset_size in [1, 3, 5, 10, 15, 20, 25, 30, 50]:
    observation_count = 10

    test_dataset = ToyDataset(test_dataset_size, observation_count, seed=2)


    def third_order_polynomial(x, a0, a1, a2, a3):
        return a0 + (a1 * x) + (a2 * x ** 2) + (a3 * x ** 3)


    errors = []
    x = np.linspace(-1, 1, num=10)
    for test_example, test_label in zip(test_dataset.examples, test_dataset.labels):
        y = test_example[:10]
        coefficients, _ = curve_fit(third_order_polynomial, x, y, bounds=([0, 1, -np.inf, -np.inf], [0 + 1e-8, 1 + 1e-8, np.inf, np.inf]))
        errors.append(coefficients[3] - test_label)
    mean_error = np.mean(np.abs(errors))
    print('{} examples: {:.5}'.format(test_dataset_size, mean_error))
