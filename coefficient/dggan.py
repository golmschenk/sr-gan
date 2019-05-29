"""Code for a dual goal regression GAN."""
from torch.nn import BCEWithLogitsLoss
import numpy as np
import torch
from scipy.stats import norm, wasserstein_distance

from coefficient.presentation import generate_display_frame
from utility import standard_image_format_to_tensorboard_image_format, gpu, MixtureModel
from coefficient.models import DgganMLP, Generator
from coefficient.srgan import CoefficientExperiment


class CoefficientDgganExperiment(CoefficientExperiment):
    """A class for an experiment of a dual goal regression GAN."""

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.DNN = DgganMLP(self.settings.hidden_size)
        self.D = DgganMLP(self.settings.hidden_size)
        self.G = Generator(self.settings.hidden_size)

    def dnn_loss_calculation(self, labeled_examples, labels):
        """Calculates the DNN loss."""
        predicted_labels, _ = self.DNN(labeled_examples)
        labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_labels, _ = self.D(labeled_examples)
        labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, labeled_examples, unlabeled_examples):
        """Calculates the unlabeled loss."""
        _, fake_scores = self.D(unlabeled_examples)
        criterion = BCEWithLogitsLoss()
        unlabeled_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
        unlabeled_loss *= self.settings.matching_loss_multiplier
        unlabeled_loss *= self.settings.dggan_loss_multiplier
        return unlabeled_loss

    def fake_loss_calculation(self, unlabeled_examples, fake_examples):
        """Calculates the fake loss."""
        _, fake_scores = self.D(fake_examples)
        criterion = BCEWithLogitsLoss()
        fake_loss = criterion(fake_scores, torch.ones_like(fake_scores))
        fake_loss *= self.settings.contrasting_loss_multiplier
        fake_loss *= self.settings.dggan_loss_multiplier
        return fake_loss

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        _, fake_scores = self.D(interpolates)
        return fake_scores

    def generator_loss_calculation(self, fake_examples, _):
        """Calculates the generator's loss."""
        _, fake_scores = self.D(fake_examples)
        criterion = BCEWithLogitsLoss()
        generator_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
        return generator_loss

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
        dnn_predicted_train_labels = DNN(torch.tensor(
            train_dataset.examples.astype(np.float32)).to(gpu))[0].to('cpu').detach().numpy()
        dnn_train_label_errors = np.mean(np.abs(dnn_predicted_train_labels - train_dataset.labels))
        dnn_summary_writer.add_scalar('2 Train Error/MAE', dnn_train_label_errors, )
        dnn_predicted_validation_labels = DNN(torch.tensor(
            validation_dataset.examples.astype(np.float32)).to(gpu))[0].to('cpu').detach().numpy()
        dnn_validation_label_errors = np.mean(np.abs(dnn_predicted_validation_labels - validation_dataset.labels))
        dnn_summary_writer.add_scalar('1 Validation Error/MAE', dnn_validation_label_errors, )
        predicted_train_labels = D(torch.tensor(
            train_dataset.examples.astype(np.float32)).to(gpu))[0].to('cpu').detach().numpy()
        gan_train_label_errors = np.mean(np.abs(predicted_train_labels - train_dataset.labels))
        gan_summary_writer.add_scalar('2 Train Error/MAE', gan_train_label_errors, )
        predicted_validation_labels = D(torch.tensor(
            validation_dataset.examples.astype(np.float32)).to(gpu))[0].to('cpu').detach().numpy()
        gan_validation_label_errors = np.mean(np.abs(predicted_validation_labels - validation_dataset.labels))
        gan_summary_writer.add_scalar('1 Validation Error/MAE', gan_validation_label_errors, )
        gan_summary_writer.add_scalar('1 Validation Error/Ratio MAE GAN DNN',
                                      gan_validation_label_errors / dnn_validation_label_errors, )
        z = torch.tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]).rvs(
            size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = G(z, add_noise=False)
        fake_examples_array = fake_examples.to('cpu').detach().numpy()
        fake_predicted_labels = D(fake_examples)[0]
        fake_predicted_labels_array = fake_predicted_labels.to('cpu').detach().numpy()
        unlabeled_labels_array = unlabeled_dataset.labels[:settings.validation_dataset_size]
        label_wasserstein_distance = wasserstein_distance(fake_predicted_labels_array, unlabeled_labels_array)
        gan_summary_writer.add_scalar('Generator/Predicted Label Wasserstein', label_wasserstein_distance, )
        unlabeled_examples_array = unlabeled_dataset.examples[:settings.validation_dataset_size]
        unlabeled_examples = torch.tensor(unlabeled_examples_array.astype(np.float32)).to(gpu)
        unlabeled_predictions = D(unlabeled_examples)[0]
        if dnn_summary_writer.step % settings.summary_step_period == 0:
            unlabeled_predictions_array = unlabeled_predictions.to('cpu').detach().numpy()
            validation_predictions_array = predicted_validation_labels
            train_predictions_array = predicted_train_labels
            dnn_validation_predictions_array = dnn_predicted_validation_labels
            dnn_train_predictions_array = dnn_predicted_train_labels
            distribution_image = generate_display_frame(fake_examples_array, unlabeled_predictions_array,
                                                        validation_predictions_array, dnn_validation_predictions_array,
                                                        train_predictions_array, dnn_train_predictions_array, step)
            distribution_image = standard_image_format_to_tensorboard_image_format(distribution_image)
            gan_summary_writer.add_image('Distributions', distribution_image)
