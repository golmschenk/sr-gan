"""Code for downloading and preprocessing the datasets."""
import os
import shutil
import zipfile
import numpy as np
from abc import abstractmethod, ABC
from urllib.request import urlretrieve

from crowd.label_generation import generate_point_density_map, generate_density_label, generate_knn_map
from utility import clean_scientific_notation


class DatabasePreprocessor(ABC):
    """A class for downloading and preprocessing the datasets"""
    def __init__(self):
        self.database_name: str = None
        self.database_url: str = None
        self.database_archived_directory_name: str = None
        self.database_directory: str
        if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
            self.database_directory = '../../{}'.format(self.database_name)
        else:
            self.database_directory = '../{}'.format(self.database_name)
        self.total_head_count = 0
        self.total_images = 0

    def download_and_preprocess(self):
        """Both download and preprocess the dataset."""
        print(f'Preparing {self.database_name} database.')
        self.download()
        self.preprocess()
        self.print_statistics()

    def download(self):
        """Downloads the dataset."""
        print('Downloading...')
        if os.path.exists(self.database_directory):
            shutil.rmtree(self.database_directory)
        os.makedirs(self.database_directory)
        temporary_archive_name = 'temporary'
        temporary_archive_path = os.path.join(self.database_directory, temporary_archive_name)
        urlretrieve(self.database_url, temporary_archive_path)
        self.extract_archive(temporary_archive_path)
        extracted_database_path = os.path.join(self.database_directory, self.database_archived_directory_name)
        files = os.listdir(extracted_database_path)
        for file_ in files:
            shutil.move(os.path.join(extracted_database_path, file_), self.database_directory)
        shutil.rmtree(extracted_database_path)
        os.remove(os.path.join(self.database_directory, temporary_archive_name))

    def extract_archive(self, temporary_archive_path: str):
        """Extracts the archive. Good for subclassing in case the archive is not zip."""
        with zipfile.ZipFile(temporary_archive_path, 'r') as zip_file:
            zip_file.extractall(self.database_directory)

    @abstractmethod
    def preprocess(self):
        """Preprocesses the labels for the dataset."""
        pass

    def generate_labels_for_example(self, dataset_directory, file_name, image, head_positions):
        """Generates the labels for a single file."""
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, -1)  # Greyscale to RGB.
        label_size = image.shape[:2]
        density_map, _ = generate_point_density_map(head_positions, label_size)
        self.total_head_count += np.sum(density_map)
        self.total_images += 1
        # Point labels.
        labels_directory = os.path.join(dataset_directory, 'labels', f'{file_name}.npy')
        os.makedirs(labels_directory, exist_ok=True)
        np.save(os.path.join(labels_directory, f'{file_name}.npy'), density_map.astype(np.float16))
        # Density labels.
        density_kernel_betas = [0.05, 0.1, 0.3, 0.5]
        for beta in density_kernel_betas:
            density_directory_name = clean_scientific_notation('density{:e}'.format(beta))
            density_directory = os.path.join(dataset_directory, density_directory_name)
            os.makedirs(density_directory, exist_ok=True)
            density_path = os.path.join(density_directory, f'{file_name}.npy')
            density_map = generate_density_label(head_positions, label_size, perspective_resizing=True,
                                                 yx_order=True, neighbor_deviation_beta=beta)
            np.save(density_path, density_map.astype(np.float16))
        # ikNN labels.
        k_list = [1, 2, 3, 4, 5]
        epsilon = 1
        for k in k_list:
            iknn_maps_directory = os.path.join(dataset_directory, f'i{k}nn_maps')
            os.makedirs(iknn_maps_directory, exist_ok=True)
            iknn_map_path = os.path.join(iknn_maps_directory, f'{file_name}.npy')
            knn_map = generate_knn_map(head_positions, label_size, number_of_neighbors=k)
            iknn_map = 1 / (knn_map + epsilon)
            np.save(iknn_map_path, iknn_map.astype(np.float16))

    def print_statistics(self):
        """Prints basic statistics for the processed database."""
        print(f'{self.total_images} images processed.')
        print(f'{self.total_head_count} head counts processed.')
