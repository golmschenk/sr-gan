"""
Code from preprocessing the UCSD dataset.
"""
import os
import random
import imageio
import numpy as np
import patoolib
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NegativeOneToOneNormalizeImage, NumpyArraysToTorchTensors
from crowd.database_preprocessor import DatabasePreprocessor
from utility import seed_all


class UcfCc50FullImageDataset(Dataset):
    """A class for the full image examples of the UCF-CC-50 crowd dataset."""
    def __init__(self, seed=None, test_start=0, dataset='train', map_directory_name='i1nn_maps'):
        seed_all(seed)
        self.dataset_directory = UcfCc50Preprocessor().database_directory
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')]
        test_file_names = self.file_names[test_start:test_start + 10]
        if dataset == 'test':
            self.file_names = test_file_names
        else:
            for file_name in test_file_names:
                self.file_names.remove(file_name)
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


class UcfCc50TransformedDataset(Dataset):
    """
    A class for the transformed UCF-CC-50 crowd dataset.
    """

    def __init__(self, image_patch_size=224, label_patch_size=224, seed=None, test_start=0, dataset='train',
                 middle_transform=None, inverse_map=False, map_directory_name='i1nn_maps'):
        seed_all(seed)
        self.dataset_directory = UcfCc50Preprocessor().database_directory
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')]
        test_file_names = self.file_names[test_start:test_start + 10]
        if dataset == 'test':
            self.file_names = test_file_names
        else:
            for file_name in test_file_names:
                self.file_names.remove(file_name)
        print('{} images.'.format(len(self.file_names)))
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        half_patch_size = int(self.image_patch_size // 2)
        self.length = 0
        self.start_indexes = []
        for file_name in self.file_names:
            self.start_indexes.append(self.length)
            image = np.load(os.path.join(self.dataset_directory, 'images', file_name))
            y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
            x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
            image_indexes_length = len(y_positions) * len(x_positions)
            self.length += image_indexes_length
        self.middle_transform = middle_transform
        self.inverse_map = inverse_map
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
        image = np.load(os.path.join(self.dataset_directory, 'images', file_name))
        label = np.load(os.path.join(self.dataset_directory, 'labels', file_name))
        map_ = np.load(os.path.join(self.dataset_directory, self.map_directory_name, file_name))
        if '1nn' in self.map_directory_name and 'i1nn' not in self.map_directory_name:
            map_ = map_ / 112
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
        map_ = example.map
        if self.inverse_map:
            map_ = 1 / (map_ + 1)
        return example.image, example.label, map_

    def __len__(self):
        return self.length


class UcfCc50Preprocessor(DatabasePreprocessor):
    """The preprocessor for the ShanghaiTech dataset."""
    def __init__(self):
        super().__init__()
        self.database_name = 'UCF CC 50'
        self.database_url = 'http://crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar'
        self.database_archived_directory_name = 'UCF_CC_50'

    def preprocess(self):
        """Preprocesses the database generating the image and map labels."""
        for mat_filename in os.listdir(self.database_directory):
            if not mat_filename.endswith('.mat'):
                continue
            file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
            mat_path = os.path.join(self.database_directory, mat_filename)
            original_image_path = os.path.join(self.database_directory, file_name + '.jpg')
            image = imageio.imread(original_image_path)
            mat = scipy.io.loadmat(mat_path)
            head_positions = mat['annPoints']  # x, y ordering.
            head_positions = self.get_y_x_head_positions(head_positions)
            self.generate_labels_for_example(self.database_directory, file_name, image, head_positions)

    @staticmethod
    def get_y_x_head_positions(original_head_positions):
        """Swaps the x's and y's of the head positions."""
        return original_head_positions[:, [1, 0]]

    def extract_archive(self, temporary_archive_path: str):
        """Extracts the archive. Used by super class."""
        patoolib.extract_archive(os.path.join(self.database_directory, temporary_archive_path),
                                 outdir=self.database_directory)


if __name__ == '__main__':
    preprocessor = UcfCc50Preprocessor()
    preprocessor.download_and_preprocess()
