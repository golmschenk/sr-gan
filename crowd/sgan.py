"""Code for the SGAN for the crowd application."""
import torch

from crowd.models import DCGenerator, JointDCDiscriminator
from crowd.srgan import CrowdExperiment
from sgan import SganExperiment
from utility import logits_to_bin_values, real_numbers_to_bin_indexes, logsumexp


class CrowdSganExperiment(SganExperiment, CrowdExperiment):
    """A class for the SGAN of the Age application."""
    def __init__(self, settings):
        super().__init__(settings)
        self.bins = torch.linspace(0, 300, settings.number_of_bins)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.G = DCGenerator()
        self.D = JointDCDiscriminator(number_of_outputs=self.settings.number_of_bins)
        self.DNN = JointDCDiscriminator(number_of_outputs=self.settings.number_of_bins)

    def images_to_predicted_labels(self, network, images):
        """Runs the code to go from images to a predicted age."""
        predicted_densities, predicted_count_logits = network(images)
        predicted_counts = logits_to_bin_values(predicted_count_logits, self.bins)
        return predicted_densities, predicted_counts

    def dnn_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predictions = self.DNN(labeled_examples)
        density_labels = labels
        predicted_density_labels, predicted_count_logits = predictions
        density_loss = torch.abs(predicted_density_labels - density_labels).pow(2).sum(1).sum(1).mean()
        count_labels = density_labels.sum(1).sum(1)
        bin_index_count_labels = real_numbers_to_bin_indexes(count_labels, self.bins)
        count_loss = self.labeled_criterion(predicted_count_logits, bin_index_count_labels)
        labeled_loss = count_loss + (density_loss * 10)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predictions = self.D(labeled_examples)
        density_labels = labels
        predicted_density_labels, predicted_count_logits = predictions
        density_loss = torch.abs(predicted_density_labels - density_labels).pow(2).sum(1).sum(1).mean()
        count_labels = density_labels.sum(1).sum(1)
        bin_index_count_labels = real_numbers_to_bin_indexes(count_labels, self.bins)
        count_loss = self.labeled_criterion(predicted_count_logits, bin_index_count_labels)
        labeled_loss = count_loss + (density_loss * 10)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, labeled_examples, unlabeled_examples):
        """Calculates the unlabeled loss."""
        _, predicted_class_logits = self.D(unlabeled_examples)
        unlabeled_binary_logits = logsumexp(predicted_class_logits, dim=1)
        ones = torch.ones_like(unlabeled_binary_logits)
        unlabeled_loss = self.gan_criterion(unlabeled_binary_logits, ones)
        unlabeled_loss *= self.settings.matching_loss_multiplier
        return unlabeled_loss

    def fake_loss_calculation(self, labeled_examples, fake_examples):
        """Calculates the fake loss."""
        _, predicted_class_logits = self.D(fake_examples.detach())
        fake_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(fake_binary_logits)
        fake_loss = self.gan_criterion(fake_binary_logits, zeros)
        fake_loss *= self.settings.matching_loss_multiplier
        return fake_loss

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        _, predicted_class_logits = self.D(interpolates)
        interpolate_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(interpolate_binary_logits)
        interpolates_loss = self.gan_criterion(interpolate_binary_logits, zeros)
        interpolates_loss *= self.settings.gradient_penalty_multiplier
        return interpolates_loss

    def generator_loss_calculation(self, fake_examples, unlabeled_examples):
        """Calculates the generator's loss."""
        _, predicted_class_logits = self.D(fake_examples)
        fake_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(fake_binary_logits)
        generator_loss = self.gan_criterion(fake_binary_logits, zeros).neg()
        return generator_loss
