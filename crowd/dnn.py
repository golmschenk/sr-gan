"""
Code for running only the DNN version of the crowd application.
"""
import random
from collections import defaultdict

import numpy as np
import torchvision
from torch.utils.data import DataLoader

from crowd import data
from crowd.data import CrowdExample
from crowd.models import DenseNetDiscriminator
from crowd.srgan import CrowdExperiment
from crowd.ucf_qnrf_data import UcfQnrfDataset
from dnn import DnnExperiment
from utility import gpu


class CrowdDnnExperiment(DnnExperiment, CrowdExperiment):
    """Runs the experiment for a DNN only version of the crowd application."""
    def dataset_setup(self):
        """Sets up the datasets for the application."""
        settings = self.settings
        if settings.crowd_dataset == 'UCF QNRF':
            self.dataset_class = UcfQnrfDataset
            patch_extractor = data.ExtractPatchForRandomPosition(self.settings.image_patch_size)
            train_transform = torchvision.transforms.Compose([patch_extractor,
                                                              data.RandomHorizontalFlip(),
                                                              data.NegativeOneToOneNormalizeImage(),
                                                              data.NumpyArraysToTorchTensors()])
            validation_transform = torchvision.transforms.Compose([patch_extractor,
                                                                   data.NegativeOneToOneNormalizeImage(),
                                                                   data.NumpyArraysToTorchTensors()])
            self.train_dataset = UcfQnrfDataset(transform=train_transform, seed=settings.labeled_dataset_seed,
                                                number_of_examples=settings.labeled_dataset_size,
                                                fake_dataset_length=True)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.validation_dataset = UcfQnrfDataset(dataset='test', transform=validation_transform, seed=101)
        else:
            raise ValueError('{} is not an understood crowd dataset.'.format(settings.crowd_dataset))

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = DenseNetDiscriminator()

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        DNN = self.DNN
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset

        self.evaluation_epoch(settings, DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer, '1 Validation Error')
        train_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities = next(train_iterator)
        dnn_predicted_densities, _ = DNN(images.to(gpu))
        dnn_real_comparison_image = self.create_crowd_images_comparison_grid(images, densities,
                                                                             dnn_predicted_densities.to('cpu'))
        dnn_summary_writer.add_image('Real', dnn_real_comparison_image)
        validation_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities = next(validation_iterator)
        dnn_predicted_densities, _ = DNN(images.to(gpu))
        dnn_validation_comparison_image = self.create_crowd_images_comparison_grid(images, densities,
                                                                                   dnn_predicted_densities.to('cpu'))
        dnn_summary_writer.add_image('Validation', dnn_validation_comparison_image)

        self.test_summaries()

    def test_summaries(self):
        """Evaluates the model on test data during training."""
        test_dataset = self.dataset_class(dataset='test')
        if self.settings.test_summary_size is not None:
            indexes = random.sample(range(test_dataset.length), self.settings.test_summary_size)
        else:
            indexes = range(test_dataset.length)
        network = self.DNN
        totals = defaultdict(lambda: 0)
        for index in indexes:
            full_image, full_label = test_dataset[index]
            full_example = CrowdExample(image=full_image, label=full_label)
            full_predicted_count, full_predicted_label = self.predict_full_example(full_example, network)
            totals['Count error'] += np.abs(full_predicted_count - full_example.label.sum())
            totals['Density sum error'] += np.abs(full_predicted_label.sum() - full_example.label.sum())
            totals['SE count'] += (full_predicted_count - full_example.label.sum()) ** 2
            totals['SE density'] += (full_predicted_label.sum() - full_example.label.sum()) ** 2
        summary_writer = self.dnn_summary_writer
        mae_count = totals['Count error'] / len(indexes)
        summary_writer.add_scalar('0 Test Error/MAE count', mae_count)
        mae_density = totals['Density sum error'] / len(indexes)
        summary_writer.add_scalar('0 Test Error/MAE density', mae_density)
        rmse_count = (totals['SE count'] / len(indexes)) ** 0.5
        summary_writer.add_scalar('0 Test Error/RMSE count', rmse_count)
        rmse_density = (totals['SE density'] / len(indexes)) ** 0.5
        summary_writer.add_scalar('0 Test Error/RMSE density', rmse_density)

    def evaluate(self, during_training=False, step=None, number_of_examples=None):
        """Evaluates the model on test data."""
        self.model_setup()
        self.load_models()
        self.gpu_mode()
        self.eval_mode()
        self.settings.dataset_class = UcfQnrfDataset
        test_dataset = self.settings.dataset_class(dataset='test')
        if self.settings.test_summary_size is not None:
            indexes = random.sample(range(test_dataset.length), self.settings.test_summary_size)
        else:
            indexes = range(test_dataset.length)
        network = self.DNN
        totals = defaultdict(lambda: 0)
        for index in indexes:
            full_image, full_label = test_dataset[index]
            full_example = CrowdExample(image=full_image, label=full_label)
            full_predicted_count, full_predicted_label = self.predict_full_example(full_example, network)
            totals['Count error'] += np.abs(full_predicted_count - full_example.label.sum())
            totals['Density sum error'] += np.abs(full_predicted_label.sum() - full_example.label.sum())
            totals['SE count'] += (full_predicted_count - full_example.label.sum()) ** 2
            totals['SE density'] += (full_predicted_label.sum() - full_example.label.sum()) ** 2
        print('Count MAE: {}'.format(totals['Count error'] / len(indexes)))
        print('Count RMSE: {}'.format((totals['SE count'] / len(indexes)) ** 0.5))
        print('Density MAE: {}'.format(totals['Density sum error'] / len(indexes)))
        print('Density RMSE: {}'.format((totals['SE density'] / len(indexes)) ** 0.5))
