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
from crowd.label_generation import generate_density_label, problematic_head_labels
from utility import seed_all

dataset_name = 'UCF QNRF'
if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../{}'.format(dataset_name)
else:
    database_directory = '../{}'.format(dataset_name)


class UcfQnrfDataset(Dataset):
    """
    A class for the UCF QNRF crowd dataset.
    """
    def __init__(self, dataset='train', transform=None, seed=None, number_of_examples=None,
                 fake_dataset_length=False):
        seed_all(seed)
        self.dataset_directory = os.path.join(database_directory, dataset.capitalize())
        self.file_names = [name for name in os.listdir(os.path.join(self.dataset_directory, 'labels'))
                           if name.endswith('.npy')][:number_of_examples]
        self.fake_dataset_length = fake_dataset_length
        self.transform = transform
        if self.transform is not None and dataset == 'test':
            self.file_names.remove('img_0100.npy')  # Image too small for non-test phase transform.
        if self.fake_dataset_length:
            self.length = int(1e6)
        else:
            self.length = len(self.file_names)

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
        """Preprocesses the database to a format with each label and image being it's own file."""
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


class UcfQnrfCheck:
    """A class for listing statistics about the UCF QNRF dataset."""
    def display_statistics(self):
        """
        Displays the statistics of the database.
        """
        print('=' * 50)
        print('UCF QNRF')
        train_dataset = UcfQnrfDataset('train')
        train_label_sums = []
        for image, label in train_dataset:
            train_label_sums.append(label.sum())
        self.print_statistics(train_label_sums, 'train')
        test_dataset = UcfQnrfDataset('test')
        test_label_sums = []
        for image, label in test_dataset:
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
    preprocessor.download()
    preprocessor.preprocess()
    UcfQnrfCheck().display_statistics()
