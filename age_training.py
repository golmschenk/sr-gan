import numpy as np

import torch
import torchvision as torchvision
from scipy.stats import norm, wasserstein_distance

from torch.autograd import Variable
from torch.utils.data import DataLoader

from age_data import AgeDataset
from age_models import Generator, Discriminator
from utility import seed_all, cpu, gpu, MixtureModel


def dataset_setup(settings):
    print('Selecting dataset...')
    seed_all(settings.labeled_dataset_seed)
    train_dataset = AgeDataset(start=0, end=settings.labeled_dataset_size)
    train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    unlabeled_dataset = AgeDataset(start=train_dataset.length,
                                   end=train_dataset.length + settings.unlabeled_dataset_size)
    unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True)
    train_and_unlabeled_dataset_size = train_dataset.length + unlabeled_dataset.length
    validation_dataset = AgeDataset(start=train_and_unlabeled_dataset_size,
                                    end=train_and_unlabeled_dataset_size + settings.validation_dataset_size)
    print('Dataset selected.')
    return train_dataset, train_dataset_loader, unlabeled_dataset, unlabeled_dataset_loader, validation_dataset


def model_setup():
    G_model = Generator()
    D_model = Discriminator()
    DNN_model = Discriminator()
    return DNN_model, D_model, G_model


def validation_summaries(D, DNN, G, dnn_summary_writer, gan_summary_writer, settings, step, train_dataset,
                         trial_directory, unlabeled_dataset, validation_dataset):
    # DNN training evaluation.
    dnn_train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size)
    dnn_train_predicted_ages, dnn_train_ages = np.array([]), np.array([])
    for images, ages in dnn_train_dataset_loader:
        predicted_ages = cpu(DNN(gpu(Variable(images))).squeeze().data).numpy()
        dnn_train_predicted_ages = np.concatenate([dnn_train_predicted_ages, predicted_ages])
        dnn_train_ages = np.concatenate([dnn_train_ages, ages])
    dnn_train_label_error = np.mean(np.abs(dnn_train_predicted_ages - dnn_train_ages))
    dnn_summary_writer.add_scalar('2 Train Error/MAE', dnn_train_label_error)
    # DNN validation evaluation.
    dnn_validation_dataset_loader = DataLoader(validation_dataset, batch_size=settings.batch_size)
    dnn_validation_predicted_ages, dnn_validation_ages = np.array([]), np.array([])
    for images, ages in dnn_validation_dataset_loader:
        predicted_ages = cpu(DNN(gpu(Variable(images))).squeeze().data).numpy()
        dnn_validation_predicted_ages = np.concatenate([dnn_validation_predicted_ages, predicted_ages])
        dnn_validation_ages = np.concatenate([dnn_validation_ages, ages])
    dnn_validation_label_error = np.mean(np.abs(dnn_validation_predicted_ages - dnn_validation_ages))
    dnn_summary_writer.add_scalar('1 Validation Error/MAE', dnn_validation_label_error)
    # GAN training evaluation.
    gan_train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size)
    gan_train_predicted_ages, gan_train_ages = np.array([]), np.array([])
    for images, ages in gan_train_dataset_loader:
        predicted_ages = cpu(D(gpu(Variable(images))).squeeze().data).numpy()
        gan_train_predicted_ages = np.concatenate([gan_train_predicted_ages, predicted_ages])
        gan_train_ages = np.concatenate([gan_train_ages, ages])
    gan_train_label_error = np.mean(np.abs(gan_train_predicted_ages - gan_train_ages))
    gan_summary_writer.add_scalar('2 Train Error/MAE', gan_train_label_error)
    # GAN validation evaluation.
    gan_validation_dataset_loader = DataLoader(validation_dataset, batch_size=settings.batch_size)
    gan_validation_predicted_ages, gan_validation_ages = np.array([]), np.array([])
    for images, ages in gan_validation_dataset_loader:
        predicted_ages = cpu(D(gpu(Variable(images))).squeeze().data).numpy()
        gan_validation_predicted_ages = np.concatenate([gan_validation_predicted_ages, predicted_ages])
        gan_validation_ages = np.concatenate([gan_validation_ages, ages])
    gan_validation_label_error = np.mean(np.abs(gan_validation_predicted_ages - gan_validation_ages))
    gan_summary_writer.add_scalar('1 Validation Error/MAE', gan_validation_label_error)
    gan_summary_writer.add_scalar('1 Validation Error/Ratio MAE GAN DNN', gan_validation_label_error / dnn_validation_label_error)
    # Real images.
    train_iterator = iter(gan_train_dataset_loader)
    examples, _ = next(train_iterator)
    images_image = torchvision.utils.make_grid(examples[:9], nrow=3)
    gan_summary_writer.add_image('Real', images_image)
    # Generated images.
    z = torch.randn(settings.batch_size, G.input_size)
    fake_examples = G(Variable(z))
    fake_images_image = torchvision.utils.make_grid(fake_examples.data[:9], nrow=3)
    gan_summary_writer.add_image('Fake/Standard', fake_images_image)
    z = torch.from_numpy(MixtureModel([norm(-settings.mean_offset, 1),
                                       norm(settings.mean_offset, 1)]
                                      ).rvs(size=[settings.batch_size, G.input_size]).astype(np.float32))
    fake_examples = G(Variable(z))
    fake_images_image = torchvision.utils.make_grid(fake_examples.data[:9], nrow=3)
    gan_summary_writer.add_image('Fake/Offset', fake_images_image)

