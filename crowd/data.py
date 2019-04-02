"""
Code for the crowd data dataset.
"""
import random
from enum import Enum
import scipy.misc
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset


class CrowdDataset(Enum):
    """An enum to select the crowd dataset."""
    ucf_cc_50 = 'UCF CC 50'
    ucf_qnrf = 'UCF QNRF'
    shanghai_tech = 'ShanghaiTech'
    world_expo = 'World Expo'


class CrowdExample:
    """A class to represent all manner of crowd example data."""
    image: np.ndarray or torch.Tensor
    label: np.ndarray or torch.Tensor or None
    roi: np.ndarray or torch.Tensor or None
    perspective: np.ndarray or torch.Tensor or None
    patch_center_y: int or None
    patch_center_x: int or None

    def __init__(self, image, label=None, roi=None, perspective=None, patch_center_y=None, patch_center_x=None,
                 map_=None):
        self.image = image
        self.label = label
        self.roi = roi
        self.map = map_
        self.perspective = perspective
        self.patch_center_y = patch_center_y
        self.patch_center_x = patch_center_x


class NumpyArraysToTorchTensors:
    """
    Converts from NumPy arrays of an example to Torch tensors.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The crowd example in Tensors.
        :rtype: CrowdExample
        """
        example.image = example.image.transpose((2, 0, 1))
        example.image = torch.tensor(example.image, dtype=torch.float32)
        if example.label is not None:
            example.label = torch.tensor(example.label, dtype=torch.float32)
        if example.map is not None:
            example.map = torch.tensor(example.map, dtype=torch.float32)
        if example.roi is not None:
            example.roi = torch.tensor(example.roi, dtype=torch.float32)
        if example.perspective is not None:
            example.perspective = torch.tensor(example.perspective, dtype=torch.float32)
        return example


class Rescale:
    """
    2D rescaling of an example (when in NumPy HWC form).
    """

    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The crowd example in Numpy with each of the arrays resized.
        :rtype: CrowdExample
        """
        example.image = scipy.misc.imresize(example.image, self.scaled_size)
        original_label_sum = np.sum(example.label)
        example.label = scipy.misc.imresize(example.label, self.scaled_size, mode='F')
        if original_label_sum != 0:
            unnormalized_label_sum = np.sum(example.label)
            example.label = (example.label / unnormalized_label_sum) * original_label_sum
        example.roi = scipy.misc.imresize(example.roi, self.scaled_size, mode='F') > 0.5
        example.map = scipy.misc.imresize(example.map, self.scaled_size, mode='F')
        return example


class RandomHorizontalFlip:
    """
    Randomly flips the example horizontally (when in NumPy HWC form).
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The possibly flipped crowd example in Numpy.
        :rtype: CrowdExample
        """
        if random.choice([True, False]):
            example.image = np.flip(example.image, axis=1).copy()
            example.label = np.flip(example.label, axis=1).copy()
            example.map = np.flip(example.map, axis=1).copy()
            if example.roi is not None:
                example.roi = np.flip(example.roi, axis=1).copy()
            if example.perspective is not None:
                example.perspective = np.flip(example.perspective, axis=1).copy()
        return example


class NegativeOneToOneNormalizeImage:
    """
    Normalizes a uint8 image to range -1 to 1.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy with image from 0 to 255.
        :type example: CrowdExample
        :return: A crowd example in NumPy with image from -1 to 1.
        :rtype: CrowdExample
        """
        example.image = (example.image.astype(np.float32) / (255 / 2)) - 1
        return example


class PatchAndRescale:
    """
    Select a patch based on a position and rescale it based on the perspective map.
    """
    def __init__(self, patch_size=128):
        self.patch_size = patch_size
        self.image_scaled_size = [self.patch_size, self.patch_size]
        self.label_scaled_size = [int(self.patch_size / 4), int(self.patch_size / 4)]

    def get_patch_for_position(self, example, y, x):
        """
        Retrieves the patch for a given position.

        :param y: The y center of the patch.
        :type y: int
        :param x: The x center of the patch.
        :type x: int
        :param example: The full example with perspective to extract the patch from.
        :type example: CrowdExample
        :return: The patch.
        :rtype: CrowdExample
        """
        patch_size_ = self.get_patch_size_for_position(example, y, x)
        half_patch_size = int(patch_size_ // 2)
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
        map_ = example.map[y - half_patch_size:y + half_patch_size,
                           x - half_patch_size:x + half_patch_size]
        return CrowdExample(image=image_patch, label=label_patch, roi=roi_patch, map_=map_)

    @staticmethod
    def get_patch_size_for_position(example, y, x):
        """
        Gets the patch size for a 3x3 meter area based of the perspective and the position.

        :param example: The example with perspective information.
        :type example: CrowdExample
        :param x: The x position of the center of the patch.
        :type x: int
        :param y: The y position of the center of the patch.
        :type y: int
        :return: The patch size.
        :rtype: float
        """
        pixels_per_meter = example.perspective[y, x]
        patch_size_ = 3 * pixels_per_meter
        return patch_size_

    @staticmethod
    def pad_example(example, y_padding=(0, 0), x_padding=(0, 0)):
        """
        Pads the example.

        :param example: The example to pad.
        :type example: CrowdExample
        :param y_padding: The amount to pad the y dimension.
        :type y_padding: (int, int)
        :param x_padding: The amount to pad the x dimension.
        :type x_padding: (int, int)
        :return: The padded example.
        :rtype: CrowdExample
        """
        z_padding = (0, 0)
        image = np.pad(example.image, (y_padding, x_padding, z_padding), 'constant')
        label = np.pad(example.label, (y_padding, x_padding), 'constant')
        roi = np.pad(example.roi, (y_padding, x_padding), 'constant', constant_values=False)
        map_ = np.pad(example.map, (y_padding, x_padding), 'constant')
        return CrowdExample(image=image, label=label, roi=roi, map_=map_)

    def resize_patch(self, patch):
        """
        :param patch: The patch to resize.
        :type patch: CrowdExample
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExample
        """
        image = scipy.misc.imresize(patch.image, self.image_scaled_size)
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        map_ = scipy.misc.imresize(patch.map, self.label_scaled_size, mode='F')
        return CrowdExample(image=image, label=label, roi=roi, map_=map_)


class ExtractPatchForPositionAndRescale(PatchAndRescale):
    """
    Given an example and a position, extracts the appropriate patch based on the perspective.
    """
    def __call__(self, example_with_perspective, y, x):
        """
        :param example_with_perspective: A crowd example with perspective.
        :type example_with_perspective: CrowdExample
        :return: A crowd example and the original patch size.
        :rtype: (CrowdExample, int)
        """
        original_patch_size = self.get_patch_size_for_position(example_with_perspective, y, x)
        patch = self.get_patch_for_position(example_with_perspective, y, x)
        roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
        patch = CrowdExample(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi, map_=patch.map)
        example = self.resize_patch(patch)
        return example, original_patch_size


class RandomlySelectPatchAndRescale(PatchAndRescale):
    """
    Selects a patch of the example and resizes it based on the perspective map.
    """
    def __call__(self, example):
        """
        :param example: A crowd example with perspective.
        :type example: CrowdExample
        :return: A crowd example.
        :rtype: CrowdExample
        """
        while True:
            y, x = self.select_random_position(example)
            patch = self.get_patch_for_position(example, y, x)
            if np.any(patch.roi):
                roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
                patch = CrowdExample(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi,
                                     map_=patch.map * patch.roi)
                example = self.resize_patch(patch)
                return example

    @staticmethod
    def select_random_position(example):
        """
        Picks a random position in the full example.

        :param example: The full example with perspective.
        :type example: CrowdExample
        :return: The y and x positions chosen randomly.
        :rtype: (int, int)
        """
        y = np.random.randint(example.label.shape[0])
        x = np.random.randint(example.label.shape[1])
        return y, x


class RandomlySelectPathWithNoPerspectiveRescale(RandomlySelectPatchAndRescale):
    """A transformation to randomly select a patch."""
    def get_patch_size_for_position(self, example, y, x):
        """
        Always returns the patch size (overriding the super class)

        :param example: The example to extract the patch from.
        :type example: ExampleNew
        :param y: The y position of the center of the patch.
        :type y: int
        :param x: The x position of the center of the patch.
        :type x: int
        :return: The size of the patch to be extracted.
        :rtype: int
        """
        return self.patch_size

    def resize_patch(self, patch):
        """
        Resizes the label and roi of the patch.

        :param patch: The patch to resize.
        :type patch: CrowdExample
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExample
        """
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        map_ = scipy.misc.imresize(patch.map, self.label_scaled_size, mode='F')
        return CrowdExample(image=patch.image, label=label, roi=roi, map_=map_)


class ExtractPatchForPositionNoPerspectiveRescale(PatchAndRescale):
    """Extracts the patch for a position."""
    def __call__(self, example_with_perspective, y, x):
        original_patch_size = self.get_patch_size_for_position(example_with_perspective, y, x)
        patch = self.get_patch_for_position(example_with_perspective, y, x)
        roi_image_patch = patch.image * np.expand_dims(patch.roi, axis=-1)
        patch = CrowdExample(image=roi_image_patch, label=patch.label * patch.roi, roi=patch.roi,
                             map_=patch.map * patch.roi)
        example = self.resize_patch(patch)
        return example, original_patch_size

    def get_patch_size_for_position(self, example, y, x):
        """
        Always returns the patch size (overriding the super class)

        :param example: The example to extract the patch from.
        :type example: ExampleWithPerspective
        :param y: The y position of the center of the patch.
        :type y: int
        :param x: The x position of the center of the patch.
        :type x: int
        :return: The size of the patch to be extracted.
        :rtype: int
        """
        return self.patch_size

    def resize_patch(self, patch):
        """
        Resizes the label and roi of the patch.

        :param patch: The patch to resize.
        :type patch: CrowdExample
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExample
        """
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.label_scaled_size, mode='F')
        unnormalized_label_sum = np.sum(label)
        if unnormalized_label_sum != 0:
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.label_scaled_size, mode='F') > 0.5
        perspective = scipy.misc.imresize(patch.perspective, self.label_scaled_size, mode='F')
        map_ = scipy.misc.imresize(patch.map, self.label_scaled_size, mode='F')
        return CrowdExample(image=patch.image, label=label, roi=roi, perspective=perspective, map_=map_)


class ExtractPatch:
    """A transform to extract a patch from an example."""
    def __init__(self, image_patch_size=128, label_patch_size=32, allow_padded=False):
        self.allow_padded = allow_padded
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size

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
        if self.allow_padded:
            if y - half_patch_size < 0:
                example = self.pad_example(example, y_padding=(half_patch_size - y, 0))
                y += half_patch_size - y
            if y + half_patch_size > example.image.shape[0]:
                example = self.pad_example(example, y_padding=(0, y + half_patch_size - example.image.shape[0]))
            if x - half_patch_size < 0:
                example = self.pad_example(example, x_padding=(half_patch_size - x, 0))
                x += half_patch_size - x
            if x + half_patch_size > example.image.shape[1]:
                example = self.pad_example(example, x_padding=(0, x + half_patch_size - example.image.shape[1]))
        else:
            assert half_patch_size <= y <= example.image.shape[0] - half_patch_size
            assert half_patch_size <= x <= example.image.shape[1] - half_patch_size
        image_patch = example.image[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size,
                                    :]
        if example.label is not None:
            label_patch = example.label[y - half_patch_size:y + half_patch_size,
                                        x - half_patch_size:x + half_patch_size]
        else:
            label_patch = None
        if example.map is not None:
            map_patch = example.map[y - half_patch_size:y + half_patch_size,
                                    x - half_patch_size:x + half_patch_size]
        else:
            map_patch = None
        if example.perspective is not None:
            perspective_patch = example.perspective[y - half_patch_size:y + half_patch_size,
                                                    x - half_patch_size:x + half_patch_size]
        else:
            perspective_patch = None
        return CrowdExample(image=image_patch, label=label_patch, perspective=perspective_patch, map_=map_patch)

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
        if example.label is not None:
            label = np.pad(example.label, (y_padding, x_padding), 'constant')
        else:
            label = None
        if example.perspective is not None:
            perspective = np.pad(example.perspective, (y_padding, x_padding), 'edge')
        else:
            perspective = None
        if example.map is not None:
            map_ = np.pad(example.map, (y_padding, x_padding), 'constant')
        else:
            map_ = None
        return CrowdExample(image=image, label=label, perspective=perspective, map_=map_)

    def resize_label(self, patch):
        """
        Resizes the label of a patch.

        :param patch: The patch.
        :type patch: CrowdExample
        :return: The patch with the resized label.
        :rtype: CrowdExample
        """
        if self.label_patch_size == self.image_patch_size:
            return patch
        label_scaled_size = [self.label_patch_size, self.label_patch_size]
        if patch.label is not None:
            original_label_sum = np.sum(patch.label)
            label = scipy.misc.imresize(patch.label, label_scaled_size, mode='F')
            unnormalized_label_sum = np.sum(label)
            if unnormalized_label_sum != 0:
                label = (label / unnormalized_label_sum) * original_label_sum
        else:
            label = None
        if patch.map is not None:
            map_ = scipy.misc.imresize(patch.map, label_scaled_size, mode='F')
        else:
            map_ = None
        if patch.perspective is not None:
            perspective = scipy.misc.imresize(patch.perspective, label_scaled_size, mode='F')
        else:
            perspective = None
        return CrowdExample(image=patch.image, label=label, perspective=perspective, map_=map_)


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

    def select_random_position(self, example):
        """
        Selects a random position from the example.

        :param example: The example.
        :type example: CrowdExample
        :return: The patch.
        :rtype: CrowdExample
        """
        if self.allow_padded:
            y = np.random.randint(example.label.shape[0])
            x = np.random.randint(example.label.shape[1])
        else:
            half_patch_size = int(self.image_patch_size // 2)
            y = np.random.randint(half_patch_size, example.label.shape[0] - half_patch_size + 1)
            x = np.random.randint(half_patch_size, example.label.shape[1] - half_patch_size + 1)
        return y, x


class ImageSlidingWindowDataset(Dataset):
    """
    Creates a database for a sliding window extraction of 1 full example (i.e. each of the patches of the full example).
    """
    def __init__(self, full_example, image_patch_size=128, window_step_size=32):
        self.full_example = CrowdExample(image=full_example.image)  # We don't need the label in this case.
        self.window_step_size = window_step_size
        self.image_patch_size = image_patch_size
        half_patch_size = int(self.image_patch_size // 2)
        self.y_positions = list(range(half_patch_size, self.full_example.image.shape[0] - half_patch_size + 1,
                                      self.window_step_size))
        if self.full_example.image.shape[0] - half_patch_size > 0:
            self.y_positions = list(set(self.y_positions + [self.full_example.image.shape[0] - half_patch_size]))
        self.x_positions = list(range(half_patch_size, self.full_example.image.shape[1] - half_patch_size + 1,
                                      self.window_step_size))
        if self.full_example.image.shape[1] - half_patch_size > 0:
            self.x_positions = list(set(self.x_positions + [self.full_example.image.shape[1] - half_patch_size]))
        self.positions_shape = np.array([len(self.y_positions), len(self.x_positions)])
        self.length = self.positions_shape.prod()

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset (the specific patch of the image).
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        extract_patch_transform = ExtractPatchForPosition(self.image_patch_size,
                                                          allow_padded=True)  # In case image is smaller than patch.
        test_transform = torchvision.transforms.Compose([NegativeOneToOneNormalizeImage(),
                                                         NumpyArraysToTorchTensors()])
        y_index, x_index = np.unravel_index(index, self.positions_shape)
        y = self.y_positions[y_index]
        x = self.x_positions[x_index]
        patch = extract_patch_transform(self.full_example, y, x)
        example = test_transform(patch)
        return example.image, x, y

    def __len__(self):
        return self.length
