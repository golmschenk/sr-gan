"""
Code from preprocessing the UCSD dataset.
"""
import os
import random
import shutil
from urllib.request import urlretrieve
import imageio
import numpy as np
import patoolib
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NegativeOneToOneNormalizeImage, NumpyArraysToTorchTensors
from crowd.label_generation import generate_point_density_map, generate_knn_map
from utility import seed_all

dataset_name = 'UCF CC 50'
if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../{}'.format(dataset_name)
else:
    database_directory = '../{}'.format(dataset_name)


class UcfCc50FullImageDataset(Dataset):
    """A class for the full image examples of the UCF-CC-50 crowd dataset."""
    def __init__(self, seed=None, test_start=0, dataset='train', map_directory_name='i1nn_maps'):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory)
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
        self.dataset_directory = os.path.join(database_directory)
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


class UcfCc50Preprocessing:
    """A class for preparing the UCF CC 50 dataset."""
    def download_and_preprocess(self):
        """Downloads and preprocesses the database."""
        print('Preparing UCF CC 50 database.')
        print('Downloading...')
        self.download()
        print('Preprocessing...')
        self.preprocess()

    @staticmethod
    def download():
        """Downloads the database."""
        if os.path.exists(database_directory):
            shutil.rmtree(database_directory)
        os.makedirs(database_directory)
        urlretrieve('http://crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar',
                    os.path.join(database_directory, 'temporary'))
        patoolib.extract_archive(os.path.join(database_directory, 'temporary'), outdir=database_directory)
        default_directory_name = 'UCF_CC_50'
        files = os.listdir(os.path.join(database_directory, default_directory_name))
        for file_ in files:
            shutil.move(os.path.join(database_directory, default_directory_name, file_), database_directory)
        shutil.rmtree(os.path.join(database_directory, default_directory_name))
        os.remove(os.path.join(database_directory, 'temporary'))

        print('Done downloading.')

    def preprocess(self):
        """Preprocesses the database to a format with each label and image being it's own file."""
        images_directory = os.path.join(database_directory, 'images')
        labels_directory = os.path.join(database_directory, 'labels')
        os.makedirs(images_directory, exist_ok=True)
        os.makedirs(labels_directory, exist_ok=True)
        for mat_filename in os.listdir(database_directory):
            if not mat_filename.endswith('.mat'):
                continue
            file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
            mat_path = os.path.join(database_directory, mat_filename)
            original_image_path = os.path.join(database_directory, file_name + '.jpg')
            image_path = os.path.join(images_directory, file_name + '.npy')
            label_path = os.path.join(labels_directory, file_name + '.npy')
            image = imageio.imread(original_image_path)
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
            label_size = image.shape[:2]
            print(label_size)
            mat = scipy.io.loadmat(mat_path)
            head_positions = mat['annPoints']  # x, y ordering.
            head_positions = self.get_y_x_head_positions(head_positions)
            for k in [1, 2, 3, 4, 5]:
                maps_directory = os.path.join(database_directory, '{}nn_maps'.format(k))
                iknn_maps_directory = os.path.join(database_directory, 'i{}nn_maps'.format(k))
                os.makedirs(maps_directory, exist_ok=True)
                os.makedirs(iknn_maps_directory, exist_ok=True)
                knn_map_path = os.path.join(maps_directory, file_name + '.npy')
                iknn_map_path = os.path.join(iknn_maps_directory, file_name + '.npy')
                knn_map = generate_knn_map(head_positions, label_size, number_of_neighbors=k, upper_bound=112)
                iknn_map = 1 / (knn_map + 1)
                np.save(iknn_map_path, iknn_map)
                np.save(knn_map_path, knn_map)
            label, _ = generate_point_density_map(head_positions, label_size)
            np.save(image_path, image)
            np.save(label_path, label)

    @staticmethod
    def get_y_x_head_positions(original_head_positions):
        """Swaps the x's and y's of the head positions."""
        return original_head_positions[:, [1, 0]]


class UcfCc50Check:
    """A class for listing statistics about the UCF CC 50 dataset."""
    @staticmethod
    def display_statistics():
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('UCF CC 50')
        dataset = UcfCc50FullImageDataset()
        label_sums = []
        for image, label in dataset:
            label_sums.append(label.sum())
        label_sums = np.array(label_sums)
        print('Person count: {}'.format(label_sums.sum()))
        print('Average count: {}'.format(label_sums.mean(axis=0)))
        print('Median count: {}'.format(np.median(label_sums, axis=0)))
        print('Max single image count: {}'.format(label_sums.max(axis=0)))
        print('Min single image count: {}'.format(label_sums.min(axis=0)))


if __name__ == '__main__':
    preprocessor = UcfCc50Preprocessing()
    preprocessor.download_and_preprocess()
