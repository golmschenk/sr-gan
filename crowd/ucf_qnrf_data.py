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
from crowd.label_generation import generate_density_label, problematic_head_labels, generate_knn_map, \
    generate_point_density_map
from utility import seed_all

dataset_name = 'UCF QNRF'
if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../{}'.format(dataset_name)
else:
    database_directory = '../{}'.format(dataset_name)


class UcfQnrfFullImageDataset(Dataset):
    """
    A class for the UCF QNRF full image crowd dataset.
    """
    def __init__(self, dataset='train', seed=None, number_of_examples=None):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory, dataset.capitalize())
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        self.length = len(self.file_names)

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
        knn_map = np.load(os.path.join(self.dataset_directory, 'knn_maps', file_name))
        return image, label, knn_map

    def __len__(self):
        return self.length


class UcfQnrfTransformedDataset(Dataset):
    """
    A class for the transformed UCF QNRF crowd dataset.
    """
    def __init__(self, dataset='train', image_patch_size=224, label_patch_size=224, seed=None, number_of_examples=None,
                 middle_transform=None):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory, dataset.capitalize())
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
        knn_map = np.load(os.path.join(self.dataset_directory, 'knn_maps', file_name), mmap_mode='r')
        half_patch_size = int(self.image_patch_size // 2)
        y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
        x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
        positions_shape = [len(y_positions), len(x_positions)]
        y_index, x_index = np.unravel_index(position_index, positions_shape)
        y = y_positions[y_index]
        x = x_positions[x_index]
        example = CrowdExample(image=image, label=label, knn_map=knn_map)
        example = extract_patch_transform(example, y, x)
        if self.middle_transform:
            example = self.middle_transform(example)
        example = preprocess_transform(example)
        return example.image, example.label, example.knn_map

    def __len__(self):
        return self.length


class UcfQnrfPreprocessing:
    """A class for preparing the UCF QNRF dataset."""
    def download_and_preprocess(self):
        """Downloads and preprocesses the database."""
        print('Preparing UCF QNRF database.')
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
        urlretrieve('http://crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip',
                    os.path.join(database_directory, 'temporary'))
        patoolib.extract_archive(os.path.join(database_directory, 'temporary'), outdir=database_directory)
        default_directory_name = 'UCF-QNRF_ECCV18'
        files = os.listdir(os.path.join(database_directory, default_directory_name))
        for file_ in files:
            shutil.move(os.path.join(database_directory, default_directory_name, file_), database_directory)
        shutil.rmtree(os.path.join(database_directory, default_directory_name))
        os.remove(os.path.join(database_directory, 'temporary'))

        print('Done downloading.')

    @staticmethod
    def preprocess():
        """Preprocesses the database to a format with each label and image being it's own file_."""
        for dataset_name_ in ['Train', 'Test']:
            images_directory = os.path.join(database_directory, dataset_name_, 'images')
            labels_directory = os.path.join(database_directory, dataset_name_, 'labels')
            os.makedirs(images_directory, exist_ok=True)
            os.makedirs(labels_directory, exist_ok=True)
            for mat_filename in os.listdir(os.path.join(database_directory, dataset_name_)):
                if not mat_filename.endswith('.mat'):
                    continue
                file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
                mat_path = os.path.join(database_directory, dataset_name_, mat_filename)
                original_image_path = os.path.join(database_directory, dataset_name_, file_name + '.jpg')
                image_path = os.path.join(images_directory, file_name + '.npy')
                label_path = os.path.join(labels_directory, file_name + '.npy')
                image = imageio.imread(original_image_path)
                if len(image.shape) == 2:
                    image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
                label_size = image.shape[:2]
                mat = scipy.io.loadmat(mat_path)
                head_positions = mat['annPoints']  # x, y ordering.
                label = generate_density_label(head_positions, label_size)
                np.save(image_path, image)
                np.save(label_path, label)
        print('Problematic head labels: {}'.format(problematic_head_labels))

    def knn_preprocess(self):
        """Generate the kNN map version of labels (along with count labels)."""
        for dataset_name_ in ['Train', 'Test']:
            images_directory = os.path.join(database_directory, dataset_name_, 'images')
            knn_maps_directory = os.path.join(database_directory, dataset_name_, 'knn_maps')
            labels_directory = os.path.join(database_directory, dataset_name_, 'labels')
            os.makedirs(images_directory, exist_ok=True)
            os.makedirs(knn_maps_directory, exist_ok=True)
            os.makedirs(labels_directory, exist_ok=True)
            for mat_filename in os.listdir(os.path.join(database_directory, dataset_name_)):
                if not mat_filename.endswith('.mat'):
                    continue
                file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
                mat_path = os.path.join(database_directory, dataset_name_, mat_filename)
                original_image_path = os.path.join(database_directory, dataset_name_, file_name + '.jpg')
                image_path = os.path.join(images_directory, file_name + '.npy')
                knn_map_path = os.path.join(knn_maps_directory, file_name + '.npy')
                label_path = os.path.join(labels_directory, file_name + '.npy')
                image = imageio.imread(original_image_path)
                if len(image.shape) == 2:
                    image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
                label_size = image.shape[:2]
                mat = scipy.io.loadmat(mat_path)
                original_head_positions = mat['annPoints']  # x, y ordering (mostly).
                # Get y, x ordering.
                head_positions = self.get_y_x_head_positions(original_head_positions, file_name, label_size)
                knn_map = generate_knn_map(head_positions, label_size, upper_bound=112)
                knn_map = knn_map.astype(np.float16)
                density_map, out_of_bounds_count = generate_point_density_map(head_positions, label_size)
                density_map = density_map.astype(np.float16)
                if density_map.sum() > 1e6:
                    print('{} is super huge.'.format(file_name))
                if out_of_bounds_count > 0:
                    print('{} has {} out of bounds.'.format(file_name, out_of_bounds_count))
                np.save(image_path, image)
                np.save(knn_map_path, knn_map)
                np.save(label_path, density_map)

    @staticmethod
    def get_y_x_head_positions(original_head_positions, file_name, label_size):
        if file_name == 'img_0087':
            # Flip y labels.
            head_position_list = []
            for original_head_position in original_head_positions:
                head_position_list.append([label_size[0] - original_head_position[0], original_head_position[1]])
            head_positions = np.array(head_position_list)
            return head_positions
        elif file_name == 'img_0006':
            # Flip x labels.
            head_position_list = []
            for original_head_position in original_head_positions:
                head_position_list.append([original_head_position[0], label_size[1] - original_head_position[1]])
            head_positions = np.array(head_position_list)
            return head_positions
        else:
            return original_head_positions[:, [1, 0]]


class UcfQnrfCheck:
    """A class for listing statistics about the UCF QNRF dataset."""
    def display_statistics(self):
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('UCF QNRF')
        train_dataset = UcfQnrfFullImageDataset('train')
        train_label_sums = []
        for image, label, knn_map in train_dataset:
            train_label_sums.append(label.sum())
        self.print_statistics(train_label_sums, 'train')
        test_dataset = UcfQnrfFullImageDataset('test')
        test_label_sums = []
        for image, label, knn_map in test_dataset:
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
    preprocessor = UcfQnrfPreprocessing()
    # preprocessor.download()
    preprocessor.knn_preprocess()
    UcfQnrfCheck().display_statistics()
