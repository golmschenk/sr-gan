import numpy as np

observation_count = 10
test_dataset_size = 10000000
test_means = np.random.normal(size=[test_dataset_size, 1])
test_standard_deviations = np.random.gamma(shape=2, size=[test_dataset_size, 1])
test_examples = np.random.normal(test_means, test_standard_deviations, size=[test_dataset_size, observation_count])
test_examples.sort(axis=1)
test_labels = np.concatenate((test_means, test_standard_deviations), axis=1)

predicted_means = test_examples.mean(axis=1)
predicted_stds = test_examples.std(axis=1)
predicted_labels = np.stack((predicted_means, predicted_stds), axis=1)

test_label_errors = np.mean(np.abs(predicted_labels - test_labels), axis=0)
print('Test Error Mean: {}'.format(test_label_errors.data[0]))
print('Test Error Std {}'.format(test_label_errors.data[1]))
