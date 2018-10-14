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
from torch.utils.data import Dataset

from crowd.data import CrowdExample
from crowd.label_generation import generate_density_label
from utility import seed_all

if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../ShanghaiTech Dataset'
else:
    database_directory = '../ShanghaiTech Dataset'


class ShanghaiTechDataset(Dataset):
    """
    A class for the ShanghaiTech crowd dataset.
    """
    def __init__(self, dataset='train', transform=None, seed=None, part='part_B', number_of_examples=None,
                 fake_dataset_length=False):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory, part, '{}_data'.format(dataset))
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        self.fake_dataset_length = fake_dataset_length
        if self.fake_dataset_length:
            self.length = int(1e6)
        else:
            self.length = len(self.file_names)
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        if self.fake_dataset_length:
            random_index = random.randrange(len(self.file_names))
            file_name = self.file_names[random_index]
        else:
            file_name = self.file_names[index]
        image = np.load(os.path.join(self.dataset_directory, 'images', file_name))
        label = np.load(os.path.join(self.dataset_directory, 'labels', file_name))
        example = CrowdExample(image=image, label=label)
        if self.transform:
            example = self.transform(example)
        return example.image, example.label

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
        """Preprocesses the database to a format with each label and image being it's own file."""
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


class ShanghaiTechCheck:
    """A class for listing statistics about the ShanghaiTech dataset."""
    def display_statistics(self):
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('part_B')
        test_dataset = ShanghaiTechDataset('test')
        test_labels = np.stack([label for (image, label) in test_dataset], axis=0)
        self.print_statistics('test', test_labels)
        train_dataset = ShanghaiTechDataset('train')
        train_labels = np.stack([label for (image, label) in train_dataset], axis=0)
        self.print_statistics('train', train_labels)
        total_dataset_name = 'total'
        total_labels = np.concatenate([test_labels, train_labels], axis=0)
        self.print_statistics(total_dataset_name, total_labels)

    @staticmethod
    def print_statistics(dataset_name, labels):
        """
        Prints the statistics for the given images and labels.

        :param dataset_name: The name of the data set being checked.
        :type dataset_name: str
        :param labels: The labels of the dataset.
        :type labels: np.ndarray
        """
        print('-' * 50)
        print('Dataset: {}'.format(dataset_name))
        print('Person count: {}'.format(labels.sum()))
        print('Average count: {}'.format(labels.sum(axis=(1, 2)).mean(axis=0)))
        print('Median count: {}'.format(np.median(labels.sum(axis=(1, 2)), axis=0)))
        print('Max single image count: {}'.format(labels.sum(axis=(1, 2)).max(axis=0)))
        print('Min single image count: {}'.format(labels.sum(axis=(1, 2)).min(axis=0)))


if __name__ == '__main__':
    preprocessor = ShanghaiTechPreprocessing()
    preprocessor.download_and_preprocess()
    # preprocessor.preprocess()
    ShanghaiTechCheck().display_statistics()
