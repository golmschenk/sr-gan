"""
Code for running only the DNN version of the crowd application.
"""
import random
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader

from crowd import data
from crowd.data import CrowdExample, CrowdDataset
from crowd.models import KnnDenseNetCat
from crowd.shanghai_tech_data import ShanghaiTechFullImageDataset, ShanghaiTechTransformedDataset
from crowd.srgan import CrowdExperiment
from crowd.ucf_cc_50_data import UcfCc50FullImageDataset, UcfCc50TransformedDataset
from crowd.ucf_qnrf_data import UcfQnrfFullImageDataset, UcfQnrfTransformedDataset
from dnn import DnnExperiment
from utility import gpu


class CrowdDnnExperiment(DnnExperiment, CrowdExperiment):
    """Runs the experiment for a DNN only version of the crowd application."""
    def dataset_setup(self):
        """Sets up the datasets for the application."""
        settings = self.settings
        if settings.crowd_dataset == CrowdDataset.ucf_qnrf:
            self.dataset_class = UcfQnrfFullImageDataset
            self.train_dataset = UcfQnrfTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                           seed=settings.labeled_dataset_seed,
                                                           map_directory_name=settings.map_directory_name,
                                                           number_of_examples=settings.labeled_dataset_size)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.validation_dataset = UcfQnrfTransformedDataset(dataset='test', seed=101,
                                                                map_directory_name=settings.map_directory_name,)
        elif settings.crowd_dataset == CrowdDataset.shanghai_tech:
            self.dataset_class = ShanghaiTechFullImageDataset
            self.train_dataset = ShanghaiTechTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                                seed=settings.labeled_dataset_seed,
                                                                number_of_examples=settings.labeled_dataset_size,
                                                                map_directory_name=settings.map_directory_name,
                                                                image_patch_size=self.settings.image_patch_size,
                                                                label_patch_size=self.settings.label_patch_size)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.validation_dataset = ShanghaiTechTransformedDataset(dataset='test', seed=101,
                                                                     map_directory_name=settings.map_directory_name,
                                                                     image_patch_size=self.settings.image_patch_size,
                                                                     label_patch_size=self.settings.label_patch_size)
        elif settings.crowd_dataset == CrowdDataset.ucf_cc_50:
            seed = 0
            self.dataset_class = UcfCc50FullImageDataset
            self.train_dataset = UcfCc50TransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                           seed=seed,
                                                           test_start=settings.labeled_dataset_seed * 10,
                                                           inverse_map=settings.inverse_map,
                                                           map_directory_name=settings.map_directory_name)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.validation_dataset = UcfCc50TransformedDataset(dataset='test', seed=seed,
                                                                test_start=settings.labeled_dataset_seed * 10,
                                                                inverse_map=settings.inverse_map,
                                                                map_directory_name=settings.map_directory_name)
        else:
            raise ValueError('{} is not an understood crowd dataset.'.format(settings.crowd_dataset))

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = KnnDenseNetCat(label_patch_size=self.settings.label_patch_size)

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        DNN = self.DNN
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset

        self.evaluation_epoch(settings, DNN, train_dataset, dnn_summary_writer, '2 Train Error', shuffle=False)
        self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer, '1 Validation Error',
                              shuffle=False)
        train_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities, maps = next(train_iterator)
        dnn_predicted_densities, _, predicted_maps = DNN(images.to(gpu))
        dnn_real_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        dnn_summary_writer.add_image('Real', dnn_real_comparison_image)
        validation_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities, maps = next(validation_iterator)
        dnn_predicted_densities, _, predicted_maps = DNN(images.to(gpu))
        dnn_validation_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        dnn_summary_writer.add_image('Validation', dnn_validation_comparison_image)

        self.test_summaries()

    def test_summaries(self):
        """Evaluates the model on test data during training."""
        test_dataset = self.dataset_class(dataset='test', map_directory_name=self.settings.map_directory_name)
        if self.settings.test_summary_size is not None:
            indexes = random.sample(range(test_dataset.length), self.settings.test_summary_size)
        else:
            indexes = range(test_dataset.length)
        network = self.DNN
        totals = defaultdict(lambda: 0)
        for index in indexes:
            full_image, full_label, full_map = test_dataset[index]
            full_example = CrowdExample(image=full_image, label=full_label)
            full_predicted_count, full_predicted_label = self.predict_full_example(full_example, network)
            totals['Count error'] += np.abs(full_predicted_count - full_example.label.sum())
            totals['NAE'] += np.abs(full_predicted_count - full_example.label.sum()) / full_example.label.sum()
            totals['Density sum error'] += np.abs(full_predicted_label.sum() - full_example.label.sum())
            totals['SE count'] += (full_predicted_count - full_example.label.sum()) ** 2
            totals['SE density'] += (full_predicted_label.sum() - full_example.label.sum()) ** 2
        summary_writer = self.dnn_summary_writer
        nae_count = totals['NAE'] / len(indexes)
        summary_writer.add_scalar('0 Test Error/NAE count', nae_count)
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
        self.settings.dataset_class = UcfQnrfFullImageDataset
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

    @property
    def inference_network(self):
        """The network to be used for inference."""
        return self.DNN
