"""Code for a regular (non-regression) GAN."""
from abc import ABC
import torch
from torch import nn

from srgan import Experiment
from utility import real_numbers_to_bin_indexes, logsumexp


class SganExperiment(Experiment, ABC):
    """A class for an experiment of a SGAN."""
    def __init__(self, settings):
        super().__init__(settings)
        self.labeled_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.bins: torch.Tensor = None

    def dnn_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        bin_index_labels = real_numbers_to_bin_indexes(labels, self.bins)
        predicted_logits = self.DNN(labeled_examples)
        labeled_loss = self.labeled_criterion(predicted_logits, bin_index_labels)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        bin_index_labels = real_numbers_to_bin_indexes(labels, self.bins)
        predicted_logits = self.D(labeled_examples)
        labeled_loss = self.labeled_criterion(predicted_logits, bin_index_labels)
        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss

    def unlabeled_loss_calculation(self, labeled_examples, unlabeled_examples):
        """Calculates the unlabeled loss."""
        predicted_class_logits = self.D(unlabeled_examples)
        unlabeled_binary_logits = logsumexp(predicted_class_logits, dim=1)
        ones = torch.ones_like(unlabeled_binary_logits)
        unlabeled_loss = self.gan_criterion(unlabeled_binary_logits, ones)
        unlabeled_loss *= self.settings.matching_loss_multiplier
        return unlabeled_loss

    def fake_loss_calculation(self, unlabeled_examples, fake_examples):
        """Calculates the fake loss."""
        predicted_class_logits = self.D(fake_examples.detach())
        fake_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(fake_binary_logits)
        fake_loss = self.gan_criterion(fake_binary_logits, zeros)
        fake_loss *= self.settings.matching_loss_multiplier
        return fake_loss

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        predicted_class_logits = self.D(interpolates)
        interpolate_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(interpolate_binary_logits)
        interpolates_loss = self.gan_criterion(interpolate_binary_logits, zeros)
        interpolates_loss *= self.settings.gradient_penalty_multiplier
        return interpolates_loss

    def generator_loss_calculation(self, fake_examples, unlabeled_examples):
        """Calculates the generator's loss."""
        predicted_class_logits = self.D(fake_examples)
        fake_binary_logits = logsumexp(predicted_class_logits, dim=1)
        zeros = torch.zeros_like(fake_binary_logits)
        generator_loss = self.gan_criterion(fake_binary_logits, zeros).neg()
        return generator_loss
