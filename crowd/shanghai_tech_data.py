"""
Code from preprocessing the UCSD dataset.
"""
import os
import random

import imageio
import numpy as np
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NumpyArraysToTorchTensors, NegativeOneToOneNormalizeImage
from crowd.database_preprocessor import DatabasePreprocessor
from utility import seed_all


class ShanghaiTechFullImageDataset(Dataset):
    """A class for the full image examples of the ShanghaiTech crowd dataset."""
    def __init__(self, dataset='train', seed=None, part='part_A', number_of_examples=None,
                 map_directory_name='knn_maps'):
        seed_all(seed)
        self.dataset_directory = os.path.join(ShanghaiTechPreprocessor().database_directory,
                                              part, '{}_data'.format(dataset))
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        self.length = len(self.file_names)
        self.map_directory_name = map_directory_name

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        file_name = self.file_names[index]
        image = np.load(os.path.join(self.dataset_directory, 'images', file_name))
        label = np.load(os.path.join(self.dataset_directory, 'labels', file_name))
        map_ = np.load(os.path.join(self.dataset_directory, self.map_directory_name, file_name))
        return image, label, map_

    def __len__(self):
        return self.length


class ShanghaiTechTransformedDataset(Dataset):
    """
    A class for the transformed ShanghaiTech crowd dataset.
    """
    def __init__(self, dataset='train', image_patch_size=224, label_patch_size=224, seed=None, part='part_A',
                 number_of_examples=None, middle_transform=None, map_directory_name='knn_maps'):
        seed_all(seed)
        self.dataset_directory = os.path.join(ShanghaiTechPreprocessor().database_directory,
                                              part, '{}_data'.format(dataset))
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        half_patch_size = int(self.image_patch_size // 2)
        self.length = 0
        self.start_indexes = []
        for file_name in self.file_names:
            self.start_indexes.append(self.length)
            image = np.load(os.path.join(self.dataset_directory, 'images', file_name), mmap_mode='r')
            y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
            x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
            image_indexes_length = len(y_positions) * len(x_positions)
            self.length += image_indexes_length
        self.middle_transform = middle_transform
        self.map_directory_name = map_directory_name

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        index_ = random.randrange(self.length)
        file_name_index = np.searchsorted(self.start_indexes, index_, side='right') - 1
        start_index = self.start_indexes[file_name_index]
        file_name = self.file_names[file_name_index]
        position_index = index_ - start_index
        extract_patch_transform = ExtractPatchForPosition(self.image_patch_size, self.label_patch_size,
                                                          allow_padded=True)  # In case image is smaller than patch.
        preprocess_transform = torchvision.transforms.Compose([NegativeOneToOneNormalizeImage(),
                                                               NumpyArraysToTorchTensors()])
        image = np.load(os.path.join(self.dataset_directory, 'images', file_name), mmap_mode='r')
        label = np.load(os.path.join(self.dataset_directory, 'labels', file_name), mmap_mode='r')
        map_ = np.load(os.path.join(self.dataset_directory, self.map_directory_name, file_name), mmap_mode='r')
        half_patch_size = int(self.image_patch_size // 2)
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


class ShanghaiTechPreprocessor(DatabasePreprocessor):
    """The preprocessor for the ShanghaiTech dataset."""
    def __init__(self):
        super().__init__()
        self.database_name = 'ShanghaiTech'
        self.database_url = 'https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=1'
        self.database_archived_directory_name = 'ShanghaiTech'

    def preprocess(self):
        """Preprocesses the database generating the image and map labels."""
        for part in ['part_A', 'part_B']:
            for dataset in ['test_data', 'train_data']:
                dataset_directory = os.path.join(self.database_directory, part, dataset)
                ground_truth_directory = os.path.join(dataset_directory, 'ground-truth')
                original_images_directory = os.path.join(dataset_directory, 'images')
                for mat_filename in os.listdir(ground_truth_directory):
                    file_name = mat_filename[3:-3]
                    mat_path = os.path.join(ground_truth_directory, mat_filename)
                    original_image_path = os.path.join(original_images_directory, file_name + 'jpg')
                    image = imageio.imread(original_image_path)
                    mat = scipy.io.loadmat(mat_path)
                    head_positions = mat['image_info'][0, 0][0][0][0]
                    head_positions = self.get_y_x_head_positions(head_positions)
                    self.generate_labels_for_example(dataset_directory, file_name, image, head_positions)

    @staticmethod
    def get_y_x_head_positions(original_head_positions):
        """Swaps the x's and y's of the head positions."""
        return original_head_positions[:, [1, 0]]


if __name__ == '__main__':
    preprocessor = ShanghaiTechPreprocessor()
    preprocessor.download_and_preprocess()
