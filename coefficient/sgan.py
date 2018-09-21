"""Code for the SGAN for the age application."""
import numpy as np
import torch

from coefficient.models import SganMLP, Generator
from coefficient.srgan import CoefficientExperiment
from sgan import SganExperiment
from utility import gpu, logits_to_bin_values


class CoefficientSganExperiment(SganExperiment, CoefficientExperiment):
    """A class for the SGAN of the coefficient application."""
    def __init__(self, settings):
        super().__init__(settings)
        self.bins = torch.linspace(-3, 3, self.settings.number_of_bins).to(gpu)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = SganMLP(self.settings.number_of_bins)
        self.D = SganMLP(self.settings.number_of_bins)
        self.G = Generator()

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        dnn_summary_writer = self.dnn_summary_writer
        gan_summary_writer = self.gan_summary_writer
        DNN = self.DNN
        D = self.D
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset

        self.evaluation_epoch(DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        dnn_validation_mae = self.evaluation_epoch(DNN, validation_dataset, dnn_summary_writer, '1 Validation Error')
        self.evaluation_epoch(D, train_dataset, gan_summary_writer, '2 Train Error')
        self.evaluation_epoch(D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_value=dnn_validation_mae)

    def evaluation_epoch(self, network, dataset, summary_writer, summary_name, comparison_value=None):
        """Runs the evaluation and summaries for the data in the dataset."""
        examples_tensor = torch.tensor(dataset.examples.astype(np.float32)).to(gpu)
        logits = network(examples_tensor)
        predicted_labels = logits_to_bin_values(logits, self.bins)
        predicted_labels = predicted_labels.to('cpu').detach().numpy()
        mae = np.mean(np.abs(predicted_labels - dataset.labels))
        summary_writer.add_scalar('{}/MAE'.format(summary_name), mae, )
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), mae / comparison_value, )
        return mae
