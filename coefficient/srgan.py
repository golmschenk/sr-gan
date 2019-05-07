"""
Code for the coefficient application.
"""
import numpy as np
import torch
from scipy.stats import norm, wasserstein_distance
from torch.utils.data import DataLoader
from recordclass import RecordClass

from srgan import Experiment
from coefficient.data import ToyDataset
from coefficient.models import Generator, MLP, observation_count
from coefficient.presentation import generate_display_frame
from utility import gpu, MixtureModel, standard_image_format_to_tensorboard_image_format


class CoefficientExperiment(Experiment):
    """The coefficient application."""

    def dataset_setup(self):
        """Sets up the datasets for the application."""
        settings = self.settings
        self.train_dataset = ToyDataset(dataset_size=settings.labeled_dataset_size, observation_count=observation_count,
                                        settings=settings, seed=settings.labeled_dataset_seed)
        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True,
                                               pin_memory=self.settings.pin_memory)
        self.unlabeled_dataset = ToyDataset(dataset_size=settings.unlabeled_dataset_size,
                                            observation_count=observation_count, settings=settings, seed=100)
        self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   pin_memory=self.settings.pin_memory)
        self.validation_dataset = ToyDataset(settings.validation_dataset_size, observation_count, seed=101,
                                             settings=settings)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = MLP(self.settings.hidden_size)
        self.D = MLP(self.settings.hidden_size)
        self.G = Generator(self.settings.hidden_size)

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        gan_summary_writer = self.gan_summary_writer
        DNN = self.DNN
        D = self.D
        G = self.G
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset
        unlabeled_dataset = self.unlabeled_dataset
        dnn_train_values = self.evaluation_epoch(DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        dnn_validation_values = self.evaluation_epoch(DNN, validation_dataset, dnn_summary_writer, '1 Validation Error')
        gan_train_values = self.evaluation_epoch(D, train_dataset, gan_summary_writer, '2 Train Error')
        gan_validation_values = self.evaluation_epoch(D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_values=dnn_validation_values)
        z = torch.tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]).rvs(
            size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = G(z, add_noise=False)
        fake_examples_array = fake_examples.to('cpu').detach().numpy()
        fake_predicted_labels = D(fake_examples)
        fake_predicted_labels_array = fake_predicted_labels.to('cpu').detach().numpy()
        unlabeled_labels_array = unlabeled_dataset.labels[:settings.validation_dataset_size]
        label_wasserstein_distance = wasserstein_distance(fake_predicted_labels_array, unlabeled_labels_array)
        gan_summary_writer.add_scalar('Generator/Predicted Label Wasserstein', label_wasserstein_distance, )
        unlabeled_examples_array = unlabeled_dataset.examples[:settings.validation_dataset_size]
        unlabeled_examples = torch.tensor(unlabeled_examples_array.astype(np.float32)).to(gpu)
        unlabeled_predictions = D(unlabeled_examples)
        if dnn_summary_writer.step % settings.summary_step_period == 0:
            unlabeled_predictions_array = unlabeled_predictions.to('cpu').detach().numpy()
            validation_predictions_array = gan_validation_values.predicted_labels
            train_predictions_array = gan_train_values.predicted_labels
            dnn_validation_predictions_array = dnn_validation_values.predicted_labels
            dnn_train_predictions_array = dnn_train_values.predicted_labels
            distribution_image = generate_display_frame(fake_examples_array, unlabeled_predictions_array,
                                                        validation_predictions_array, dnn_validation_predictions_array,
                                                        train_predictions_array, dnn_train_predictions_array, step)
            distribution_image = standard_image_format_to_tensorboard_image_format(distribution_image)
            gan_summary_writer.add_image('Distributions', distribution_image)

    class ComparisonValues(RecordClass):
        """A record class to hold the names of values which might be compared among methods."""
        mae: float
        mse: float
        rmse: float
        predicted_labels: np.ndarray

    def evaluation_epoch(self, network, dataset: ToyDataset, summary_writer, summary_name: str,
                         comparison_values: ComparisonValues = None):
        """An evaluation of the dataset writing to TensorBoard."""
        predicted_labels = network(torch.tensor(dataset.examples.astype(np.float32)).to(gpu)).to('cpu').detach().numpy()
        mae = np.mean(np.abs(predicted_labels - dataset.labels))
        summary_writer.add_scalar(f'{summary_name}/MAE', mae)
        mse = np.mean(np.power(predicted_labels - dataset.labels, 2))
        summary_writer.add_scalar(f'{summary_name}/MSE', mse)
        rmse = np.power(mse, 0.5)
        summary_writer.add_scalar(f'{summary_name}/RMSE', rmse)
        if comparison_values:
            summary_writer.add_scalar(f'{summary_name}/Ratio MAE GAN DNN', mae / comparison_values.mae)
            summary_writer.add_scalar(f'{summary_name}/Ratio MSE GAN DNN', mae / comparison_values.mse)
            summary_writer.add_scalar(f'{summary_name}/Ratio RMSE GAN DNN', rmse / comparison_values.rmse)
        return self.ComparisonValues(mae=mae, mse=mse, rmse=rmse, predicted_labels=predicted_labels)
