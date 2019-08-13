"""
Code for the World Expo dataset.
"""
import json
import math
import os
import random
from typing import Dict, List

import numpy as np
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NegativeOneToOneNormalizeImage, NumpyArraysToTorchTensors
from utility import seed_all


database_directory = '../World Expo'


class CameraData:
    """The data for a given camera as part of the dataset."""
    def __init__(self, images, labels, roi, perspective):
        self.images = images
        self.labels = labels
        self.roi = roi
        self.perspective = perspective


class WorldExpoFullImageDataset(Dataset):
    """
    A class for the World Expo full image crowd dataset.
    """
    def __init__(self, dataset='train', seed=None, number_of_cameras=None, number_of_images_per_camera=None,
                 map_directory_name=None):
        seed_all(seed)
        self.dataset_directory = database_directory
        with open(os.path.join(self.dataset_directory, 'viable_with_validation_and_random_test.json')) as json_file:
            cameras_dict = json.load(json_file)
        camera_names = cameras_dict[dataset]
        random.shuffle(camera_names)
        self.camera_data_list: List[CameraData] = []
        camera_names = camera_names[:number_of_cameras]
        self.length = 0
        self.start_indexes = []
        for camera_name in camera_names:
            camera_directory = os.path.join(self.dataset_directory, camera_name)
            if dataset == 'unlabeled':
                camera_images = np.load(os.path.join(camera_directory, 'unlabeled_images.npy'), mmap_mode='r')
                camera_labels = None
            else:
                camera_images = np.load(os.path.join(camera_directory, 'images.npy'), mmap_mode='r')
                camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'), mmap_mode='r')
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'), mmap_mode='r')
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'), mmap_mode='r')
            permutation_indexes = np.random.permutation(camera_images.shape[0])
            camera_images = camera_images[permutation_indexes][:number_of_images_per_camera]
            if dataset != 'unlabeled':
                camera_labels = camera_labels[permutation_indexes][:number_of_images_per_camera]
            self.camera_data_list.append(CameraData(images=camera_images, labels=camera_labels, roi=camera_roi,
                                                    perspective=camera_perspective))
            self.start_indexes.append(self.length)
            self.length += camera_images.shape[0]

    def __getitem__(self, index):
        index_ = random.randrange(self.length)
        camera_data_index = np.searchsorted(self.start_indexes, index_, side='right') - 1
        start_index = self.start_indexes[camera_data_index]
        camera_data = self.camera_data_list[camera_data_index]
        position_index = index_ - start_index
        image = camera_data.images[position_index]
        label = camera_data.labels[position_index]
        map_ = label
        return image, label, map_

    def __len__(self):
        return self.length


class WorldExpoTransformedDataset(Dataset):
    """
    A class for the transformed World Expo crowd dataset.
    """
    def __init__(self, dataset='train', image_patch_size=224, label_patch_size=224, seed=None, number_of_cameras=None,
                 number_of_images_per_camera=None, middle_transform=None):
        seed_all(seed)
        self.dataset_directory = database_directory
        with open(os.path.join(self.dataset_directory, 'viable_with_validation_and_random_test.json')) as json_file:
            cameras_dict = json.load(json_file)
        camera_names = cameras_dict[dataset]
        random.shuffle(camera_names)
        self.camera_data_list: List[CameraData] = []
        camera_names = camera_names[:number_of_cameras]
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        half_patch_size = int(self.image_patch_size // 2)
        self.length = 0
        self.start_indexes = []
        for camera_name in camera_names:
            camera_directory = os.path.join(self.dataset_directory, camera_name)
            if dataset == 'unlabeled':
                camera_images = np.load(os.path.join(camera_directory, 'unlabeled_images.npy'), mmap_mode='r')
                camera_labels = None
            else:
                camera_images = np.load(os.path.join(camera_directory, 'images.npy'), mmap_mode='r')
                camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'), mmap_mode='r')
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'), mmap_mode='r')
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'), mmap_mode='r')
            permutation_indexes = np.random.permutation(camera_images.shape[0])
            camera_images = camera_images[permutation_indexes][:number_of_images_per_camera]
            if dataset != 'unlabeled':
                camera_labels = camera_labels[permutation_indexes][:number_of_images_per_camera]
            self.camera_data_list.append(CameraData(images=camera_images, labels=camera_labels, roi=camera_roi,
                                                    perspective=camera_perspective))
            y_positions = range(half_patch_size, camera_images.shape[1] - half_patch_size + 1)
            x_positions = range(half_patch_size, camera_images.shape[2] - half_patch_size + 1)
            image_indexes_length = len(y_positions) * len(x_positions)
            self.start_indexes.append(self.length)
            self.length += camera_images.shape[0] * image_indexes_length
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        self.middle_transform = middle_transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        index_ = random.randrange(self.length)
        camera_data_index = np.searchsorted(self.start_indexes, index_, side='right') - 1
        start_index = self.start_indexes[camera_data_index]
        camera_data = self.camera_data_list[camera_data_index]
        camera_images = camera_data.images
        array_index = index_ - start_index
        half_patch_size = int(self.image_patch_size // 2)
        y_positions = range(half_patch_size, camera_images.shape[1] - half_patch_size + 1)
        x_positions = range(half_patch_size, camera_images.shape[2] - half_patch_size + 1)
        image_indexes_length = len(y_positions) * len(x_positions)
        image_index = math.floor(array_index / image_indexes_length)
        position_index = array_index % image_indexes_length
        image = camera_data.images[image_index]
        label = camera_data.labels[image_index]
        map_ = label
        extract_patch_transform = ExtractPatchForPosition(self.image_patch_size, self.label_patch_size,
                                                          allow_padded=True)  # In case image is smaller than patch.
        preprocess_transform = torchvision.transforms.Compose([NegativeOneToOneNormalizeImage(),
                                                               NumpyArraysToTorchTensors()])
        y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
        x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
        positions_shape = [len(y_positions), len(x_positions)]
        y_index, x_index = np.unravel_index(position_index, positions_shape)
        y = y_positions[y_index]
        x = x_positions[x_index]
        example = CrowdExample(image=image, label=label, map_=map_)
        example = extract_patch_transform(example, y, x)
        if self.middle_transform:
            example = self.middle_transform(example)
        example = preprocess_transform(example)
        return example.image, example.label, example.map

    def __len__(self):
        return self.length
