"""
Code for the crowd data dataset.
"""
import random
import scipy.misc

import torch
from collections import namedtuple
import os
import numpy as np
from torch.utils.data import Dataset

from utility import seed_all

patch_size = 128


CrowdExampleWithPerspective = namedtuple('CrowdExampleWithPerspective', ['image', 'label', 'roi', 'perspective'])
CrowdExampleWithRoi = namedtuple('CrowdExampleWithRoi', ['image', 'label', 'roi'])
CrowdExample = namedtuple('CrowdExampleWithRoi', ['image', 'label'])
CrowdExampleWithPosition = namedtuple('CrowdExampleWithPosition', ['image', 'label', 'x', 'y'])


class CrowdDataset(Dataset):
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


class NumpyArraysToTorchTensors:
    """
    Converts from NumPy arrays of an example to Torch tensors.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExampleWithRoi or CrowdExample
        :return: The crowd example in Tensors.
        :rtype: CrowdExampleWithRoi or CrowdExample
        """
        image = example.image.transpose((2, 0, 1))
        image = torch.tensor(image)
        label = torch.tensor(example.label)
        if isinstance(example, CrowdExampleWithRoi):
            roi = torch.tensor(example.roi.astype(np.float32))
            return CrowdExampleWithRoi(image=image, label=label, roi=roi)
        else:
            return CrowdExample(image=image, label=label)


class Rescale:
    """
    2D rescaling of an example (when in NumPy HWC form).
    """

    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExampleWithRoi
        :return: The crowd example in Numpy with each of the arrays resized.
        :rtype: CrowdExampleWithRoi
        """
        image = scipy.misc.imresize(example.image, self.scaled_size)
        original_label_sum = np.sum(example.label)
        label = scipy.misc.imresize(example.label, self.scaled_size, mode='F')
        if original_label_sum != 0:
            unnormalized_label_sum = np.sum(label)
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(example.roi, self.scaled_size, mode='F') > 0.5
        return CrowdExampleWithRoi(image=image, label=label, roi=roi)


class RandomHorizontalFlip:
    """
    Randomly flips the example horizontally (when in NumPy HWC form).
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExampleWithRoi
        :return: The possibly flipped crowd example in Numpy.
        :rtype: CrowdExampleWithRoi
        """
        if random.choice([True, False]):
            image = np.flip(example.image, axis=1).copy()
            label = np.flip(example.label, axis=1).copy()
            if isinstance(example, CrowdExampleWithRoi):
                roi = np.flip(example.roi, axis=1).copy()
                return CrowdExampleWithRoi(image=image, label=label, roi=roi)
            else:
                return CrowdExample(image=image, label=label)
        else:
            return example


class NegativeOneToOneNormalizeImage:
    """
    Normalizes a uint8 image to range -1 to 1.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy with image from 0 to 255.
        :type example: CrowdExampleWithRoi
        :return: A crowd example in NumPy with image from -1 to 1.
        :rtype: CrowdExampleWithRoi
        """
        image = (example.image.astype(np.float32) / (255 / 2)) - 1
        if isinstance(example, CrowdExampleWithRoi):
            return CrowdExampleWithRoi(image=image, label=example.label, roi=example.roi)
        else:
            return CrowdExample(image=image, label=example.label)


class PatchAndRescale:
    """
    Select a patch based on a position and rescale it based on the perspective map.
    """
    def __init__(self):
        self.image_scaled_size = [patch_size, patch_size]
        self.label_scaled_size = [int(patch_size / 4), int(patch_size / 4)]

    def get_patch_for_position(self, example_with_perspective, y, x):
        """
        Retrieves the patch for a given position.

        :param y: The y center of the patch.
        :type y: int
        :param x: The x center of the patch.
        :type x: int
        :param example_with_perspective: The full example with perspective to extract the patch from.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: The patch.
        :rtype: CrowdExampleWithRoi
        """
        patch_size_ = self.get_patch_size_for_position(example_with_perspective, y, x)
        half_patch_size = int(patch_size_ // 2)
        example = CrowdExampleWithRoi(image=example_with_perspective.image, label=example_with_perspective.label,
                                      roi=example_with_perspective.roi)
        if y - half_patch_size < 0:
            example = self.pad_example(example, y_padding=(half_patch_size - y, 0))
            y += half_patch_size - y
        if y + half_patch_size > example.label.shape[0]:
            example = self.pad_example(example, y_padding=(0, y + half_patch_size - example.label.shape[0]))
        if x - half_patch_size < 0:
            example = self.pad_example(example, x_padding=(half_patch_size - x, 0))
            x += half_patch_size - x
        if x + half_patch_size > example.label.shape[1]:
            example = self.pad_example(example, x_padding=(0, x + half_patch_size - example.label.shape[1]))
        image_patch = example.image[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size,
                                    :]
        label_patch = example.label[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size]
        roi_patch = example.roi[y - half_patch_size:y + half_patch_size,
                                x - half_patch_size:x + half_patch_size]
        return CrowdExampleWithRoi(image=image_patch, label=label_patch, roi=roi_patch)

    @staticmethod
    def get_patch_size_for_position(example_with_perspective, y, x):
        """
        Gets the patch size for a 3x3 meter area based of the perspective and the position.

        :param example_with_perspective: The example with perspective information.
        :type example_with_perspective: CrowdExampleWithPerspective
        :param x: The x position of the center of the patch.
        :type x: int
        :param y: The y position of the center of the patch.
        :type y: int
        :return: The patch size.
        :rtype: float
        """
        pixels_per_meter = example_with_perspective.perspective[y, x]
        patch_size_ = 3 * pixels_per_meter
        return patch_size_

    @staticmethod
    def pad_example(example, y_padding=(0, 0), x_padding=(0, 0)):
        """
        Pads the example.

        :param example: The example to pad.
        :type example: CrowdExampleWithRoi
        :param y_padding: The amount to pad the y dimension.
        :type y_padding: (int, int)
        :param x_padding: The amount to pad the x dimension.
        :type x_padding: (int, int)
        :return: The padded example.
        :rtype: CrowdExampleWithRoi
        """
        z_padding = (0, 0)
        image = np.pad(example.image, (y_padding, x_padding, z_padding), 'constant')
        label = np.pad(example.label, (y_padding, x_padding), 'constant')
        roi = np.pad(example.roi, (y_padding, x_padding), 'constant', constant_values=False)
        return CrowdExampleWithRoi(image=image, label=label, roi=roi)

    def resize_patch(self, patch):
        """
        :param patch: The patch to resize.
        :type patch: CrowdExampleWithRoi
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExampleWithRoi
        """
        image = scipy.misc.imresize(patch.image, self.image_scaled_size)
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        return CrowdExampleWithRoi(image=image, label=label, roi=roi)


class ExtractPatchForPositionAndRescale(PatchAndRescale):
    """
    Given an example and a position, extracts the appropriate patch based on the perspective.
    """
    def __call__(self, example_with_perspective, y, x):
        """
        :param example_with_perspective: A crowd example with perspective.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: A crowd example and the original patch size.
        :rtype: (CrowdExampleWithRoi, int)
        """
        original_patch_size = self.get_patch_size_for_position(example_with_perspective, y, x)
        patch = self.get_patch_for_position(example_with_perspective, y, x)
        roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
        patch = CrowdExampleWithRoi(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi)
        example = self.resize_patch(patch)
        return example, original_patch_size


class RandomlySelectPatchAndRescale(PatchAndRescale):
    """
    Selects a patch of the example and resizes it based on the perspective map.
    """
    def __call__(self, example_with_perspective):
        """
        :param example_with_perspective: A crowd example with perspective.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: A crowd example.
        :rtype: CrowdExampleWithRoi
        """
        while True:
            y, x = self.select_random_position(example_with_perspective)
            patch = self.get_patch_for_position(example_with_perspective, y, x)
            if np.any(patch.roi):
                roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
                patch = CrowdExampleWithRoi(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi)
                example = self.resize_patch(patch)
                return example

    @staticmethod
    def select_random_position(example_with_perspective):
        """
        Picks a random position in the full example.

        :param example_with_perspective: The full example with perspective.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: The y and x positions chosen randomly.
        :rtype: (int, int)
        """
        y = np.random.randint(example_with_perspective.label.shape[0])
        x = np.random.randint(example_with_perspective.label.shape[1])
        return y, x


class RandomlySelectPathWithNoPerspectiveRescale(RandomlySelectPatchAndRescale):
    """A transformation to randomly select a patch."""
    @staticmethod
    def get_patch_size_for_position(example_with_perspective, y, x):
        """
        Always returns the patch size (overriding the super class)

        :param example_with_perspective: The example to extract the patch from.
        :type example_with_perspective: ExampleWithPerspective
        :param y: The y position of the center of the patch.
        :type y: int
        :param x: The x position of the center of the patch.
        :type x: int
        :return: The size of the patch to be extracted.
        :rtype: int
        """
        return patch_size

    def resize_patch(self, patch):
        """
        Resizes the label and roi of the patch.

        :param patch: The patch to resize.
        :type patch: CrowdExampleWithRoi
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExampleWithRoi
        """
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        return CrowdExampleWithRoi(image=patch.image, label=label, roi=roi)


class ExtractPatchForPositionNoPerspectiveRescale(PatchAndRescale):
    """Extracts the patch for a position."""
    def __call__(self, example_with_perspective, y, x):
        original_patch_size = self.get_patch_size_for_position(example_with_perspective, y, x)
        patch = self.get_patch_for_position(example_with_perspective, y, x)
        roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
        patch = CrowdExampleWithRoi(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi)
        example = self.resize_patch(patch)
        return example, original_patch_size

    @staticmethod
    def get_patch_size_for_position(example_with_perspective, y, x):
        """
        Always returns the patch size (overriding the super class)

        :param example_with_perspective: The example to extract the patch from.
        :type example_with_perspective: ExampleWithPerspective
        :param y: The y position of the center of the patch.
        :type y: int
        :param x: The x position of the center of the patch.
        :type x: int
        :return: The size of the patch to be extracted.
        :rtype: int
        """
        return patch_size

    def resize_patch(self, patch):
        """
        Resizes the label and roi of the patch.

        :param patch: The patch to resize.
        :type patch: CrowdExampleWithRoi
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExampleWithRoi
        """
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        return CrowdExampleWithRoi(image=patch.image, label=label, roi=roi)


class ExtractPatch():
    """A transform to extract a patch from an example."""
    def __init__(self):
        self.image_patch_size = patch_size
        self.label_scaled_size = [int(patch_size / 4), int(patch_size / 4)]

    def get_patch_for_position(self, example, y, x):
        """
        Extracts a patch for a given position.

        :param example: The example to extract the patch from.
        :type example: CrowdExample
        :param y: The y position of the center of the patch.
        :type y: int
        :param x: The x position of the center of the patch.
        :type x: int
        :return: The patch.
        :rtype: CrowdExample
        """
        half_patch_size = int(self.image_patch_size // 2)
        if y - half_patch_size < 0:
            example = self.pad_example(example, y_padding=(half_patch_size - y, 0))
            y += half_patch_size - y
        if y + half_patch_size > example.label.shape[0]:
            example = self.pad_example(example, y_padding=(0, y + half_patch_size - example.label.shape[0]))
        if x - half_patch_size < 0:
            example = self.pad_example(example, x_padding=(half_patch_size - x, 0))
            x += half_patch_size - x
        if x + half_patch_size > example.label.shape[1]:
            example = self.pad_example(example, x_padding=(0, x + half_patch_size - example.label.shape[1]))
        image_patch = example.image[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size,
                                    :]
        label_patch = example.label[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size]
        return CrowdExample(image=image_patch, label=label_patch)

    @staticmethod
    def pad_example(example, y_padding=(0, 0), x_padding=(0, 0)):
        """
        Pads the given example.

        :param example: The example to pad.
        :type example: CrowdExample
        :param y_padding: The amount to pad the y axis by.
        :type y_padding: (int, int)
        :param x_padding: The amount to pad the x axis by.
        :type x_padding: (int, int)
        :return: The padded example.
        :rtype: CrowdExample
        """
        z_padding = (0, 0)
        image = np.pad(example.image, (y_padding, x_padding, z_padding), 'constant')
        label = np.pad(example.label, (y_padding, x_padding), 'constant')
        return CrowdExample(image=image, label=label)

    def resize_label(self, patch):
        """
        Resizes the label of a patch.

        :param patch: The patch.
        :type patch: CrowdExample
        :return: The patch with the resized label.
        :rtype: CrowdExample
        """
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        return CrowdExample(image=patch.image, label=label)


class ExtractPatchForPosition(ExtractPatch):
    """A transform to extract a patch for a give position."""
    def __call__(self, example, y, x):
        patch = self.get_patch_for_position(example, y, x)
        example = self.resize_label(patch)
        return example


class ExtractPatchForRandomPosition(ExtractPatch):
    """A transform to extract a patch for a random position."""
    def __call__(self, example):
        y, x = self.select_random_position(example)
        patch = self.get_patch_for_position(example, y, x)
        example = self.resize_label(patch)
        return example

    @staticmethod
    def select_random_position(example):
        """
        Selects a random position from the example.

        :param example: The example.
        :type example: CrowdExample
        :return: The patch.
        :rtype: CrowdExample
        """
        y = np.random.randint(example.label.shape[0])
        x = np.random.randint(example.label.shape[1])
        return y, x
