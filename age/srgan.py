"""Code for the age estimation application."""
import numpy as np
import torch
import torchvision as torchvision
from scipy.stats import norm
from torch.utils.data import DataLoader

from age.data import AgeDataset
from age.models import Generator, Discriminator
from age.vgg import vgg16
from srgan import Experiment
from utility import seed_all, gpu, MixtureModel, to_image_range

model_architecture = 'dcgan'  # dcgan or vgg


class AgeExperiment(Experiment):
    """The age estimation application."""
    def dataset_setup(self):
        """Sets up the datasets for the application."""
        if model_architecture == 'vgg':
            dataset_path = '../imdb_wiki_data/imdb_preprocessed_224'
        else:
            dataset_path = '../imdb_wiki_data/imdb_preprocessed_128'
        settings = self.settings
        seed_all(settings.labeled_dataset_seed)
        self.train_dataset = AgeDataset(dataset_path, start=0, end=settings.labeled_dataset_size,
                                        seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)
        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True,
                                               pin_memory=self.settings.pin_memory,
                                               num_workers=settings.number_of_data_workers, drop_last=True)
        self.unlabeled_dataset = AgeDataset(dataset_path, start=settings.labeled_dataset_size,
                                            end=settings.labeled_dataset_size + settings.unlabeled_dataset_size,
                                            seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)
        self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers, drop_last=True)
        self.validation_dataset = AgeDataset(dataset_path, start=-settings.validation_dataset_size,
                                             end=None,
                                             seed=settings.labeled_dataset_seed, batch_size=settings.batch_size)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        if model_architecture == 'vgg':
            self.G = Generator(image_size=224)
            self.D = vgg16(pretrained=True, num_classes=1)
            self.DNN = vgg16(pretrained=True, num_classes=1)
        else:
            self.G = Generator()
            self.D = Discriminator()
            self.DNN = Discriminator()

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
        # DNN training evaluation.
        self.evaluation_epoch(settings, DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        # DNN validation evaluation.
        dnn_validation_mae = self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer,
                                                   '1 Validation Error')
        # GAN training evaluation.
        self.evaluation_epoch(settings, D, train_dataset, gan_summary_writer, '2 Train Error')
        # GAN validation evaluation.
        self.evaluation_epoch(settings, D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_value=dnn_validation_mae)
        # Real images.
        train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
        train_iterator = iter(train_dataset_loader)
        examples, _ = next(train_iterator)
        images_image = torchvision.utils.make_grid(to_image_range(examples[:9]), normalize=True, range=(0, 255), nrow=3)
        gan_summary_writer.add_image('Real', images_image.numpy())
        # Generated images.
        z = torch.randn(settings.batch_size, G.input_size).to(gpu)
        fake_examples = G(z).to('cpu')
        fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), normalize=True,
                                                        range=(0, 255), nrow=3)
        gan_summary_writer.add_image('Fake/Standard', fake_images_image.numpy())
        z = torch.as_tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]
                                         ).rvs(size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = G(z).to('cpu')
        fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), normalize=True,
                                                        range=(0, 255), nrow=3)
        gan_summary_writer.add_image('Fake/Offset', fake_images_image.numpy())

    def evaluation_epoch(self, settings, network, dataset, summary_writer, summary_name, comparison_value=None):
        """Runs the evaluation and summaries for the data in the dataset."""
        dataset_loader = DataLoader(dataset, batch_size=settings.batch_size)
        predicted_ages, ages = np.array([]), np.array([])
        for images, labels in dataset_loader:
            batch_predicted_ages = self.images_to_predicted_ages(network, images.to(gpu))
            batch_predicted_ages = batch_predicted_ages.detach().to('cpu').view(-1).numpy()
            ages = np.concatenate([ages, labels])
            predicted_ages = np.concatenate([predicted_ages, batch_predicted_ages])
        mae = np.abs(predicted_ages - ages).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), mae, )
        mse = (np.abs(predicted_ages - ages) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), mse, )
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), mae / comparison_value, )
        return mae

    def images_to_predicted_ages(self, network, images):
        """Runs the code to go from images to a predicted age. Useful for overriding in subclasses."""
        predicted_ages = network(images)
        return predicted_ages
