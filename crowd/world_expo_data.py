import os
import random

import numpy as np
from torch.utils.data import Dataset

from crowd.data import CrowdExampleWithPerspective
from utility import seed_all


class WorldExpoDataset(Dataset):
    """
    A class for the crowd dataset.
    """
    def __init__(self, dataset_directory, camera_names, number_of_cameras=None, number_of_images_per_camera=None,
                 transform=None, seed=None, unlabeled=False):
        seed_all(seed)
        random.shuffle(camera_names)
        cameras_images = []
        cameras_labels = []
        cameras_rois = []
        cameras_perspectives = []
        for camera_name in camera_names[:number_of_cameras]:
            camera_directory = os.path.join(dataset_directory, camera_name)
            if unlabeled:
                camera_images = np.load(os.path.join(camera_directory, 'unlabeled_images.npy'), mmap_mode='r')
                camera_labels = np.zeros(camera_images.shape[:3], dtype=np.float32)
            else:
                camera_images = np.load(os.path.join(camera_directory, 'images.npy'), mmap_mode='r')
                camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'), mmap_mode='r')
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'), mmap_mode='r')
            camera_rois = np.repeat(camera_roi[np.newaxis, :, :], camera_images.shape[0], axis=0)
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'), mmap_mode='r')
            camera_perspectives = np.repeat(camera_perspective[np.newaxis, :, :], camera_images.shape[0], axis=0)
            permutation_indexes = np.random.permutation(len(camera_labels))
            cameras_images.append(camera_images[permutation_indexes][:number_of_images_per_camera])
            cameras_labels.append(camera_labels[permutation_indexes][:number_of_images_per_camera])
            cameras_rois.append(camera_rois[permutation_indexes][:number_of_images_per_camera])
            cameras_perspectives.append(camera_perspectives[permutation_indexes][:number_of_images_per_camera])
        self.images = np.concatenate(cameras_images, axis=0)
        self.labels = np.concatenate(cameras_labels, axis=0)
        self.rois = np.concatenate(cameras_rois, axis=0)
        self.perspectives = np.concatenate(cameras_perspectives, axis=0)
        self.length = self.labels.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        example = CrowdExampleWithPerspective(image=self.images[index], label=self.labels[index], roi=self.rois[index],
                                              perspective=self.perspectives[index])
        if self.transform:
            example = self.transform(example)
        return example.image, example.label

    def __len__(self):
        return self.length