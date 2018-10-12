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
from torch.utils.data import Dataset

from crowd.data import CrowdExample
from crowd.label_generation import generate_density_label
from utility import seed_all

dataset_name = 'UCF CC 50'
if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../{}'.format(dataset_name)
else:
    database_directory = '../{}'.format(dataset_name)


class UcfCc50Dataset(Dataset):
    """
    A class for the UCF CC 50 crowd dataset.
    """
    def __init__(self, transform=None, seed=None, number_of_examples=None,
                 fake_dataset_length=False):
        seed_all(seed)
        self.dataset_directory = database_directory
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        if fake_dataset_length:
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
        example_index = random.randrange(len(self.file_names))
        file_name = self.file_names[example_index]
        image = np.load(os.path.join(self.dataset_directory, 'images', file_name))
        label = np.load(os.path.join(self.dataset_directory, 'labels', file_name))
        example = CrowdExample(image=image, label=label)
        if self.transform:
            example = self.transform(example)
        return example.image, example.label

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

    @staticmethod
    def preprocess():
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
            mat = scipy.io.loadmat(mat_path)
            head_positions = mat['annPoints']  # x, y ordering.
            label = generate_density_label(head_positions, label_size)
            np.save(image_path, image)
            np.save(label_path, label)


class UcfCc50Check:
    """A class for listing statistics about the UCF CC 50 dataset."""
    def display_statistics(self):
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('UCF CC 50')
        dataset = UcfCc50Dataset()
        labels = [label for (image, label) in dataset]
        self.print_statistics(labels)

    @staticmethod
    def print_statistics(labels):
        """
        Prints the statistics for the given images and labels.

        :param dataset_name_: The name of the data set being checked.
        :type dataset_name_: str
        :param labels: The labels of the dataset.
        :type labels: list[np.ndarray]
        """
        print('-' * 50)
        print('Person count: {}'.format(np.array([label.sum() for label in labels]).sum()))
        print('Average count: {}'.format(np.array([label.sum(axis=(1, 2)) for label in labels]).mean(axis=0)))
        print('Median count: {}'.format(np.median(np.array([label.sum(axis=(1, 2)) for label in labels]), axis=0)))
        print('Max single image count: {}'.format(np.array([label.sum(axis=(1, 2)) for label in labels]).max(axis=0)))
        print('Min single image count: {}'.format(np.array([label.sum(axis=(1, 2)) for label in labels]).min(axis=0)))


if __name__ == '__main__':
    preprocessor = UcfCc50Preprocessing()
    preprocessor.download_and_preprocess()
    # preprocessor.preprocess()
    UcfCc50Check().display_statistics()
