"""
Code from preprocessing the UCSD dataset.
"""
import os
import random
import shutil
import zipfile
from urllib.request import urlretrieve

import imageio
import numpy as np
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NumpyArraysToTorchTensors, NegativeOneToOneNormalizeImage
from crowd.label_generation import generate_density_label, generate_knn_map, generate_point_density_map
from utility import seed_all, clean_scientific_notation

if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../ShanghaiTech Dataset'
else:
    database_directory = '../ShanghaiTech Dataset'


class ShanghaiTechFullImageDataset(Dataset):
    """A class for the full image examples of the ShanghaiTech crowd dataset."""
    def __init__(self, dataset='train', seed=None, part='part_A', number_of_examples=None,
                 map_directory_name='knn_maps'):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory, part, '{}_data'.format(dataset))
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
        self.dataset_directory = os.path.join(database_directory, part, '{}_data'.format(dataset))
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


class ShanghaiTechPreprocessing:
    """A class for preparing the ShanghaiTech dataset."""
    def download_and_preprocess(self):
        """Downloads and preprocesses the database."""
        print('Preparing ShanghaiTech database.')
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
        urlretrieve('https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=1',
                    os.path.join(database_directory, 'temporary'))
        with zipfile.ZipFile(os.path.join(database_directory, 'temporary'), 'r') as zip_file:
            zip_file.extractall(database_directory)
        files = os.listdir(os.path.join(database_directory, 'ShanghaiTech'))
        for file_ in files:
            shutil.move(os.path.join(database_directory, 'ShanghaiTech', file_), database_directory)
        shutil.rmtree(os.path.join(database_directory, 'ShanghaiTech'))
        os.remove(os.path.join(database_directory, 'temporary'))

        print('Done downloading.')

    @staticmethod
    def preprocess():
        """Preprocesses the database to a format with each label and image being it's own file_."""
        for part in ['part_A', 'part_B']:
            for dataset in ['test_data', 'train_data']:
                ground_truth_directory = os.path.join(database_directory, part, dataset, 'ground-truth')
                images_directory = os.path.join(database_directory, part, dataset, 'images')
                labels_directory = os.path.join(database_directory, part, dataset, 'labels')
                os.makedirs(labels_directory, exist_ok=True)
                for mat_filename in os.listdir(ground_truth_directory):
                    file_name = mat_filename[3:-3]
                    mat_path = os.path.join(ground_truth_directory, mat_filename)
                    original_image_path = os.path.join(images_directory, file_name + 'jpg')
                    image_path = os.path.join(images_directory, file_name + 'npy')
                    label_path = os.path.join(labels_directory, file_name + 'npy')
                    image = imageio.imread(original_image_path)
                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
                    label_size = image.shape[:2]
                    mat = scipy.io.loadmat(mat_path)
                    head_positions = mat['image_info'][0, 0][0][0][0]
                    label = generate_density_label(head_positions, label_size)
                    np.save(image_path, image)
                    np.save(label_path, label)

    def knn_preprocess(self):
        """Generate the kNN map version of labels (along with count labels)."""
        for part in ['part_A', 'part_B']:
            for dataset_name_ in ['train_data', 'test_data']:
                ground_truth_directory = os.path.join(database_directory, part, dataset_name_, 'ground-truth')
                images_directory = os.path.join(database_directory, part, dataset_name_, 'images')
                labels_directory = os.path.join(database_directory, part, dataset_name_, 'labels')
                os.makedirs(images_directory, exist_ok=True)
                os.makedirs(labels_directory, exist_ok=True)
                for mat_filename in os.listdir(ground_truth_directory):
                    if not mat_filename.endswith('.mat'):
                        continue
                    file_name = mat_filename[3:-3]
                    mat_path = os.path.join(ground_truth_directory, mat_filename)
                    original_image_path = os.path.join(images_directory, file_name + 'jpg')
                    image_path = os.path.join(images_directory, file_name + 'npy')
                    label_path = os.path.join(labels_directory, file_name + 'npy')
                    image = imageio.imread(original_image_path)
                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
                    label_size = image.shape[:2]
                    mat = scipy.io.loadmat(mat_path)
                    head_positions = mat['image_info'][0, 0][0][0][0]
                    head_positions = self.get_y_x_head_positions(head_positions)
                    for k in [1, 2, 3, 4, 5, 6]:
                        knn_maps_directory = os.path.join(database_directory, part, dataset_name_,
                                                          '{}nn_maps'.format(k))
                        iknn_maps_directory = os.path.join(database_directory, part, dataset_name_,
                                                           'i{}nn_maps'.format(k))
                        os.makedirs(knn_maps_directory, exist_ok=True)
                        os.makedirs(iknn_maps_directory, exist_ok=True)
                        knn_map_path = os.path.join(knn_maps_directory, file_name + 'npy')
                        iknn_map_path = os.path.join(iknn_maps_directory, file_name + 'npy')
                        knn_map = generate_knn_map(head_positions, label_size, number_of_neighbors=k, upper_bound=112)
                        iknn_map = 1 / (knn_map + 1)
                        np.save(iknn_map_path, iknn_map.astype(np.float16))
                        np.save(knn_map_path, knn_map.astype(np.float16))
                    density_map, out_of_bounds_count = generate_point_density_map(head_positions, label_size)
                    if out_of_bounds_count > 0:
                        print('{} has {} out of bounds.'.format(file_name, out_of_bounds_count))
                    np.save(image_path, image)
                    np.save(label_path, density_map.astype(np.float16))

    def density_preprocess(self):
        """Generate various versions of density labels with different Gaussian spread parameters."""
        density_kernel_betas = [0.05, 0.1, 0.3, 0.5]
        for part in ['part_A', 'part_B']:
            for dataset_name_ in ['train_data', 'test_data']:
                ground_truth_directory = os.path.join(database_directory, part, dataset_name_, 'ground-truth')
                images_directory = os.path.join(database_directory, part, dataset_name_, 'images')
                os.makedirs(images_directory, exist_ok=True)
                for mat_filename in os.listdir(ground_truth_directory):
                    if not mat_filename.endswith('.mat'):
                        continue
                    file_name = mat_filename[3:-3]
                    mat_path = os.path.join(ground_truth_directory, mat_filename)
                    original_image_path = os.path.join(images_directory, file_name + 'jpg')
                    image = imageio.imread(original_image_path)
                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
                    label_size = image.shape[:2]
                    mat = scipy.io.loadmat(mat_path)
                    head_positions = mat['image_info'][0, 0][0][0][0]
                    head_positions = self.get_y_x_head_positions(head_positions)
                    for density_kernel_beta in density_kernel_betas:
                        density_directory_name = clean_scientific_notation('density{:e}'.format(density_kernel_beta))
                        density_directory = os.path.join(database_directory, part, dataset_name_,
                                                         density_directory_name)
                        os.makedirs(density_directory, exist_ok=True)
                        density_path = os.path.join(density_directory, file_name + 'npy')
                        density_map = generate_density_label(head_positions, label_size, perspective_resizing=True,
                                                             yx_order=True, neighbor_deviation_beta=density_kernel_beta)
                        np.save(density_path, density_map.astype(np.float16))

    @staticmethod
    def get_y_x_head_positions(original_head_positions):
        """Swaps the x's and y's of the head positions."""
        return original_head_positions[:, [1, 0]]


class ShanghaiTechCheck:
    """Provides a brief analysis of the data after being preprocessed."""
    def display_statistics(self):
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('ShanghaiTech')
        train_dataset = ShanghaiTechFullImageDataset('train')
        train_label_sums = []
        for image, label, map_ in train_dataset:
            train_label_sums.append(label.sum())
        self.print_statistics(train_label_sums, 'train')
        test_dataset = ShanghaiTechFullImageDataset('test')
        test_label_sums = []
        for image, label, map_ in test_dataset:
            test_label_sums.append(label.sum())
        self.print_statistics(test_label_sums, 'test')
        self.print_statistics(train_label_sums + test_label_sums, 'total')

    @staticmethod
    def print_statistics(label_sums, dataset_name_):
        """
        Prints the statistics for the given images and labels.

        :param dataset_name_: The name of the data set being checked.
        :type dataset_name_: str
        :param label_sums: The sums of the labels of the dataset.
        :type label_sums: list[float]
        """
        print('-' * 50)
        print(dataset_name_)
        label_sums = np.array(label_sums)
        print('Person count: {}'.format(label_sums.sum()))
        print('Average count: {}'.format(label_sums.mean(axis=0)))
        print('Median count: {}'.format(np.median(label_sums, axis=0)))
        print('Max single image count: {}'.format(label_sums.max(axis=0)))
        print('Min single image count: {}'.format(label_sums.min(axis=0)))


if __name__ == '__main__':
    preprocessor = ShanghaiTechPreprocessing()
    # preprocessor.download_and_preprocess()
    preprocessor.download()
    preprocessor.knn_preprocess()
    preprocessor.density_preprocess()
