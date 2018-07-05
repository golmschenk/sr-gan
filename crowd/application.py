import json
import scipy.misc
import matplotlib
import numpy as np
import os
import torch
from scipy.stats import norm
import torchvision
from torch.utils.data import DataLoader

from crowd import data
from crowd.data import CrowdDataset, resized_patch_size
from application import Application
from crowd.models import DCGenerator, JointDCDiscriminator
from utility import gpu, to_image_range, MixtureModel


class CrowdApplication(Application):
    def dataset_setup(self, experiment):
        train_transform = torchvision.transforms.Compose([data.RandomlySelectPatchAndRescale(),
                                                          data.RandomHorizontalFlip(),
                                                          data.NegativeOneToOneNormalizeImage(),
                                                          data.NumpyArraysToTorchTensors()])
        validation_transform = torchvision.transforms.Compose([data.RandomlySelectPatchAndRescale(),
                                                               data.NegativeOneToOneNormalizeImage(),
                                                               data.NumpyArraysToTorchTensors()])
        settings = experiment.settings
        dataset_path = '/media/root/Gold/crowd/data/World Expo/'
        with open(os.path.join(dataset_path, 'viable_with_validation_and_random_test.json')) as json_file:
            cameras_dict = json.load(json_file)
        train_dataset = CrowdDataset(dataset_path, camera_names=cameras_dict['train'], number_of_cameras=experiment.settings.number_of_cameras,
                                     number_of_images_per_camera=experiment.settings.number_of_images_per_camera, transform=train_transform, seed=settings.labeled_dataset_seed)
        train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, pin_memory=True,
                                          num_workers=2)
        unlabeled_dataset = CrowdDataset(dataset_path, camera_names=cameras_dict['validation'], transform=train_transform, unlabeled=True)
        unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True,
                                              pin_memory=True, num_workers=2)
        validation_dataset = CrowdDataset(dataset_path, camera_names=cameras_dict['validation'], transform=validation_transform)
        return train_dataset, train_dataset_loader, unlabeled_dataset, unlabeled_dataset_loader, validation_dataset

    def model_setup(self):
        G_model = DCGenerator()
        D_model = JointDCDiscriminator()
        DNN_model = JointDCDiscriminator()
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
        dnn_validation_count_mae = self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer,
                                                         '1 Validation Error')
        # GAN training evaluation.
        self.evaluation_epoch(settings, D, train_dataset, gan_summary_writer, '2 Train Error')
        # GAN validation evaluation.
        self.evaluation_epoch(settings, D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_value=dnn_validation_count_mae)
        # Real images.
        train_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        examples, densities = next(train_iterator)
        predicted_densities, _ = D(examples.to(gpu))
        real_comparison_image = self.create_crowd_images_comparison_grid(examples.to('cpu'), densities.to('cpu'),
                                                                         predicted_densities.to('cpu'))
        gan_summary_writer.add_image('Real', real_comparison_image)
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
        predicted_counts, densities, predicted_densities = np.array([]), np.array(
            []), np.array([])
        for images, labels in dataset_loader:
            batch_predicted_densities, batch_predicted_counts = network(images.to(gpu))
            batch_predicted_densities = batch_predicted_densities.detach().to('cpu').numpy()
            batch_predicted_counts = batch_predicted_counts.detach().to('cpu').numpy()
            predicted_counts = np.concatenate([predicted_counts, batch_predicted_counts])
            if predicted_densities.size == 0:
                predicted_densities = predicted_densities.reshape([0, *batch_predicted_densities.shape[1:]])
            predicted_densities = np.concatenate([predicted_densities, batch_predicted_densities])
            if densities.size == 0:
                densities = densities.reshape([0, *labels.shape[1:]])
            densities = np.concatenate([densities, labels])
        count_mae = np.abs(predicted_counts - densities.sum(1).sum(1)).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), count_mae)
        density_mae = np.abs(predicted_densities - densities).sum(1).sum(1).mean()
        summary_writer.add_scalar('{}/Density MAE'.format(summary_name), density_mae)
        count_mse = (np.abs(predicted_counts - densities.sum(1).sum(1)) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), count_mse)
        density_mse = (np.abs(predicted_densities - densities) ** 2).sum(1).sum(1).mean()
        summary_writer.add_scalar('{}/Density MSE'.format(summary_name), density_mse)
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), count_mae / comparison_value)
        return count_mae

    def convert_density_maps_to_heatmaps(self, label, predicted_label):
        """
        Converts a label and predicted label density map into their respective heatmap images.

        :param label: The label tensor.
        :type label: torch.autograd.Variable
        :param predicted_label: The predicted labels tensor.
        :type predicted_label: torch.autograd.Variable
        :return: The heatmap label tensor and heatmap predicted label tensor.
        :rtype: (torch.autograd.Variable, torch.autograd.Variable)
        """
        mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
        label_array = label.numpy()
        predicted_label_array = predicted_label.numpy()
        mappable.set_clim(vmin=min(label_array.min(), predicted_label_array.min()),
                          vmax=max(label_array.max(), predicted_label_array.max()))
        resized_label_array = scipy.misc.imresize(label_array, (resized_patch_size, resized_patch_size), mode='F')
        label_heatmap_array = mappable.to_rgba(resized_label_array).astype(np.float32)
        label_heatmap_tensor = torch.from_numpy(label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
        resized_predicted_label_array = scipy.misc.imresize(predicted_label_array, (resized_patch_size, resized_patch_size), mode='F')
        predicted_label_heatmap_array = mappable.to_rgba(resized_predicted_label_array).astype(np.float32)
        predicted_label_heatmap_tensor = torch.from_numpy(predicted_label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
        return label_heatmap_tensor, predicted_label_heatmap_tensor

    def create_crowd_images_comparison_grid(self, images, labels, predicted_labels, number_of_images=3):
        """
        Creates a grid of images from the original images, the true labels, and the predicted labels.

        :param images: The original RGB images.
        :type images: torch.autograd.Variable
        :param labels: The labels.
        :type labels: torch.autograd.Variable
        :param predicted_labels: The predicted labels.
        :type predicted_labels: torch.autograd.Variable
        :param number_of_images: The number of (original) images to include in the grid.
        :type number_of_images: int
        :return: The image of the grid of images.
        :rtype: np.ndarray
        """
        grid_image_list = []
        for index in range(min(number_of_images, images.size()[0])):
            grid_image_list.append((images[index].data + 1) / 2)
            label_heatmap, predicted_label_heatmap = self.convert_density_maps_to_heatmaps(labels[index].data,
                                                                                           predicted_labels[index].data)
            grid_image_list.append(label_heatmap)
            grid_image_list.append(predicted_label_heatmap)
        return torchvision.utils.make_grid(grid_image_list, nrow=number_of_images)

    def labeled_loss_function(self, predicted_labels, labels, order=2):
        density_labels = labels
        predicted_density_labels, predicted_count_labels = predicted_labels
        density_loss = torch.abs(predicted_density_labels - density_labels).pow(2).sum(1).sum(1).mean()
        count_loss = torch.abs(predicted_count_labels - density_labels.sum(1).sum(1)).pow(2).mean()
        return count_loss + (density_loss * 10)
