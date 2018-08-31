"""Code for the SGAN for the ImageNet application."""
import torch
import torchvision

from age.models import Generator, Discriminator
from sgan import SganExperiment
from utility import logits_to_bin_values, seed_all


class ImageNetSganExperiment(SganExperiment):
    """A class for the SGAN of the Age application."""
    def __init__(self, settings):
        super().__init__(settings)
        self.bins = torch.linspace(10, 95, settings.number_of_bins)

    def dataset_setup(self):
        """Sets up the datasets for the application."""
        settings = self.settings
        seed_all(settings.labeled_dataset_seed)
        self.train_dataset = AgeDataset(dataset_path, start=0, end=settings.labeled_dataset_size)
        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=settings.number_of_data_workers)
        self.unlabeled_dataset = AgeDataset(dataset_path, start=self.train_dataset.length,
                                       end=self.train_dataset.length + settings.unlabeled_dataset_size)
        self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=settings.number_of_data_workers)
        train_and_unlabeled_dataset_size = self.train_dataset.length + self.unlabeled_dataset.length
        self.validation_dataset = AgeDataset(dataset_path, start=train_and_unlabeled_dataset_size,
                                             end=train_and_unlabeled_dataset_size + settings.validation_dataset_size)

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
