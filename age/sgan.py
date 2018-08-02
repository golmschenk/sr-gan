"""Code for the SGAN for the age application."""
import torch

from age.models import Generator, Discriminator
from age.srgan import AgeExperiment
from sgan import SganExperiment
from utility import logits_to_bin_values


class AgeSganExperiment(SganExperiment, AgeExperiment):
    """A class for the SGAN of the Age application."""
    def __init__(self, settings):
        super().__init__(settings)
        self.bins = torch.linspace(10, 95, settings.number_of_bins)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.G = Generator()
        self.D = Discriminator(number_of_outputs=10)
        self.DNN = Discriminator(number_of_outputs=10)

    def images_to_predicted_ages(self, network, images):
        """Runs the code to go from images to a predicted age."""
        predicted_logits = network(images)
        predicted_ages = logits_to_bin_values(predicted_logits, self.bins)
        return predicted_ages
