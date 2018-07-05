
import numpy as np
import torch
import torchvision as torchvision
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from age.data import AgeDataset
from age.models import Generator, Discriminator
from application import Application
from utility import seed_all, gpu, MixtureModel, to_image_range

model_architecture = 'dcgan'  # dcgan or vgg


class AgeApplication(Application):
    def dataset_setup(self, experiment):
        if model_architecture == 'vgg':
            dataset_path = '../imdb_wiki_data/imdb_preprocessed_256'
        else:
            dataset_path = '../imdb_wiki_data/imdb_preprocessed_128'
        settings = experiment.settings
        seed_all(settings.labeled_dataset_seed)
        train_dataset = AgeDataset(dataset_path, start=0, end=settings.labeled_dataset_size)
        train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        unlabeled_dataset = AgeDataset(dataset_path, start=train_dataset.length,
                                       end=train_dataset.length + settings.unlabeled_dataset_size)
        unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        train_and_unlabeled_dataset_size = train_dataset.length + unlabeled_dataset.length
        validation_dataset = AgeDataset(dataset_path, start=train_and_unlabeled_dataset_size,
                                        end=train_and_unlabeled_dataset_size + settings.validation_dataset_size)
        return train_dataset, train_dataset_loader, unlabeled_dataset, unlabeled_dataset_loader, validation_dataset


    def model_setup(self):
        if model_architecture == 'vgg':
            G_model = Generator(image_size=256)
            D_model = vgg16(num_classes=1)
            DNN_model = vgg16(num_classes=1)
        else:
            G_model = Generator()
            D_model = Discriminator()
            DNN_model = Discriminator()
        return DNN_model, D_model, G_model


    def validation_summaries(self, experiment, step):
        settings = experiment.settings
        dnn_summary_writer = experiment.dnn_summary_writer
        gan_summary_writer = experiment.gan_summary_writer
        DNN = experiment.DNN
        D = experiment.D
        G = experiment.G
        train_dataset = experiment.train_dataset
        validation_dataset = experiment.validation_dataset
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
        train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size)
        train_iterator = iter(train_dataset_loader)
        examples, _ = next(train_iterator)
        images_image = torchvision.utils.make_grid(to_image_range(examples[:9]), nrow=3)
        gan_summary_writer.add_image('Real', images_image.numpy().transpose([1, 2, 0]).astype(np.uint8))
        # Generated images.
        z = torch.randn(settings.batch_size, G.input_size).to(gpu)
        fake_examples = G(z).to('cpu')
        fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), nrow=3)
        gan_summary_writer.add_image('Fake/Standard', fake_images_image.numpy().transpose([1, 2, 0]).astype(np.uint8))
        z = torch.from_numpy(MixtureModel([norm(-settings.mean_offset, 1),
                                           norm(settings.mean_offset, 1)]
                                          ).rvs(size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = G(z).to('cpu')
        fake_images_image = torchvision.utils.make_grid(to_image_range(fake_examples.data[:9]), nrow=3)
        gan_summary_writer.add_image('Fake/Offset', fake_images_image.numpy().transpose([1, 2, 0]).astype(np.uint8))


    def evaluation_epoch(self, settings, network, dataset, summary_writer, summary_name, comparison_value=None):
        dataset_loader = DataLoader(dataset, batch_size=settings.batch_size)
        predicted_ages, ages = np.array([]), np.array([])
        for images, labels in dataset_loader:
            batch_predicted_ages = network(images.to(gpu))
            batch_predicted_ages = batch_predicted_ages.detach().to('cpu').numpy()
            ages = np.concatenate([ages, labels])
            predicted_ages = np.concatenate([predicted_ages, batch_predicted_ages])
        mae = np.abs(predicted_ages - ages).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), mae)
        mse = (np.abs(predicted_ages - ages) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), mse)
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), mae / comparison_value)
        return mae