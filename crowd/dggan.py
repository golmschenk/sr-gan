"""Code for a dual goal regression GAN."""
from torch.nn import BCEWithLogitsLoss
import torch

from crowd.models import KnnDenseNetCatDggan, DCGenerator, DenseNetDiscriminatorDggan
from crowd.srgan import CrowdExperiment


class CrowdDgganExperiment(CrowdExperiment):
    """A class for the DGGAN crowd experiment."""
    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.G = DCGenerator()
        self.D = KnnDenseNetCatDggan()
        self.DNN = KnnDenseNetCatDggan()

    def unlabeled_loss_calculation(self, labeled_examples, unlabeled_examples):
        """Calculates the unlabeled loss."""
        _ = self.D(unlabeled_examples)
        fake_scores = self.D.real_label
        criterion = BCEWithLogitsLoss()
        unlabeled_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
        unlabeled_loss *= self.settings.matching_loss_multiplier
        unlabeled_loss *= self.settings.dggan_loss_multiplier
        return unlabeled_loss

    def fake_loss_calculation(self, unlabeled_examples, fake_examples):
        """Calculates the fake loss."""
        _ = self.D(fake_examples)
        fake_scores = self.D.real_label
        criterion = BCEWithLogitsLoss()
        fake_loss = criterion(fake_scores, torch.ones_like(fake_scores))
        fake_loss *= self.settings.contrasting_loss_multiplier
        fake_loss *= self.settings.dggan_loss_multiplier
        return fake_loss

    def interpolate_loss_calculation(self, interpolates):
        """Calculates the interpolate loss for use in the gradient penalty."""
        _ = self.D(interpolates)
        fake_scores = self.D.real_label
        return fake_scores

    def generator_loss_calculation(self, fake_examples, _):
        """Calculates the generator's loss."""
        _ = self.D(fake_examples)
        fake_scores = self.D.real_label
        criterion = BCEWithLogitsLoss()
        generator_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
        return generator_loss
