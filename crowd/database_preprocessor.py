"""Code for downloading and preprocessing the datasets."""
import os
import shutil
import zipfile
import numpy as np
from abc import abstractmethod, ABC
from urllib.request import urlretrieve

from sklearn.neighbors import NearestNeighbors
from utility import clean_scientific_notation


class DatabasePreprocessor(ABC):
    """A class for downloading and preprocessing the datasets"""
    def __init__(self):
        self.database_name: str = None
        self.database_url: str = None
        self.database_archived_directory_name: str = None
        self.total_head_count = 0
        self.total_images = 0

    @property
    def database_directory(self):
        """The location of the database directory."""
        if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
            return '../../{}'.format(self.database_name)
        else:
            return '../{}'.format(self.database_name)

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
        # Images as numpy.
        images_directory = os.path.join(dataset_directory, 'images', f'{file_name}.npy')
        os.makedirs(images_directory, exist_ok=True)
        np.save(os.path.join(images_directory, f'{file_name}.npy'), image)
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


head_standard_deviation_meters = 0.2
body_width_standard_deviation_meters = 0.2
body_height_standard_deviation_meters = 0.5
body_height_offset_meters = 0.875


def generate_density_label(head_positions, label_size, perspective=None, include_body=False, ignore_tiny=False,
                           force_full_image_count_normalize=True, perspective_resizing=True, yx_order=False,
                           neighbor_deviation_beta=0.15):
    """
    Generates a density label given the head positions and other meta data.

    :param perspective_resizing: Marks whether any form of perspective head sizing should be used.
    :type perspective_resizing: bool
    :param force_full_image_count_normalize: Whether to normalize the label sum value to the head count.
    :type force_full_image_count_normalize: bool
    :param ignore_tiny: Whether the label should exclude annotations for positions with very small perspective values.
    :type ignore_tiny: bool
    :param include_body: Whether the label should add a body annotation.
    :type include_body: bool
    :param label_size: The dimensions the generated label should have.
    :type label_size: (int, int)
    :param head_positions: The head labeling positions.
    :type head_positions: np.ndarray
    :param perspective: The perspective map.
    :type perspective: np.ndarray
    :param yx_order: Marks whether the image is in yx order or xy order.
    :type yx_order: bool
    :param neighbor_deviation_beta: The gaussian spread parameter beta to multiply the nearest neighbor distance by.
    :type neighbor_deviation_beta: float
    :return: The density labeling.
    :rtype: np.ndarray
    """
    head_count = 0
    body_parts = 2 if include_body else 1
    number_of_neighbors = min(11, len(head_positions))
    nearest_neighbors = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='ball_tree').fit(head_positions)
    neighbor_distances, _ = nearest_neighbors.kneighbors(head_positions)
    average_neighbor_distances = neighbor_distances[0:].mean(axis=1)
    label = np.zeros(shape=label_size, dtype=np.float32)
    for head_index, head_position in enumerate(head_positions):
        if yx_order:
            y, x = np.rint(head_position).astype(np.uint32)
        else:
            x, y = np.rint(head_position).astype(np.uint32)
        if perspective_resizing:
            if perspective is not None:
                if 0 <= x < perspective.shape[1]:
                    position_perspective = perspective[y, x]
                else:
                    position_perspective = perspective[y, 0]
                if ignore_tiny and position_perspective < 3.1:
                    continue
                head_standard_deviation = position_perspective * head_standard_deviation_meters
            else:
                # This is the method used in the MCNN paper (or at least a close approximation).
                head_standard_deviation = average_neighbor_distances[head_index] * neighbor_deviation_beta
                position_perspective = None
        else:
            head_standard_deviation = 8
            position_perspective = None
        head_gaussian = make_gaussian(head_standard_deviation)
        head_gaussian = head_gaussian / (body_parts * head_gaussian.sum())
        head_count += 1
        person_label = np.zeros_like(label, dtype=np.float32)
        off_center_size = int((head_gaussian.shape[0] - 1) / 2)
        y_start_offset = 0
        if y - off_center_size < 0:
            y_start_offset = off_center_size - y
        y_end_offset = 0
        if y + off_center_size >= person_label.shape[0]:
            y_end_offset = (y + off_center_size + 1) - person_label.shape[0]
        x_start_offset = 0
        if x - off_center_size < 0:
            x_start_offset = off_center_size - x
        x_end_offset = 0
        if x + off_center_size >= person_label.shape[1]:
            x_end_offset = (x + off_center_size + 1) - person_label.shape[1]
        y_out_of_bounds = head_gaussian.shape[0] <= max(y_start_offset, y_end_offset)
        x_out_of_bounds = head_gaussian.shape[1] <= max(x_start_offset, x_end_offset)
        if y_out_of_bounds or x_out_of_bounds:
            print('Offset out of head gaussian bounds. Skipping person.')
            continue
        person_label[y - off_center_size + y_start_offset:y + off_center_size + 1 - y_end_offset,
                     x - off_center_size + x_start_offset:x + off_center_size + 1 - x_end_offset
                     ] += head_gaussian[y_start_offset:head_gaussian.shape[0] - y_end_offset,
                                        x_start_offset:head_gaussian.shape[1] - x_end_offset]
        if perspective is not None and include_body:
            body_x = x
            body_y = y + int(position_perspective * body_height_offset_meters)
            body_width_standard_deviation = position_perspective * body_width_standard_deviation_meters
            body_height_standard_deviation = position_perspective * body_height_standard_deviation_meters
            body_gaussian = make_gaussian((body_width_standard_deviation, body_height_standard_deviation))
            body_gaussian = body_gaussian / (body_parts * body_gaussian.sum())
            x_off_center_size = int((body_gaussian.shape[1] - 1) / 2)
            y_off_center_size = int((body_gaussian.shape[0] - 1) / 2)
            y_start_offset = 0
            if body_y - y_off_center_size < 0:
                y_start_offset = y_off_center_size - body_y
            y_end_offset = 0
            if body_y + y_off_center_size >= person_label.shape[0]:
                y_end_offset = (body_y + y_off_center_size + 1) - person_label.shape[0]
            x_start_offset = 0
            if body_x - x_off_center_size < 0:
                x_start_offset = x_off_center_size - body_x
            x_end_offset = 0
            if body_x + x_off_center_size >= person_label.shape[1]:
                x_end_offset = (body_x + x_off_center_size + 1) - person_label.shape[1]
            person_label[body_y - y_off_center_size + y_start_offset:body_y + y_off_center_size + 1 - y_end_offset,
                         body_x - x_off_center_size + x_start_offset:body_x + x_off_center_size + 1 - x_end_offset
                         ] += body_gaussian[y_start_offset:body_gaussian.shape[0] - y_end_offset,
                                            x_start_offset:body_gaussian.shape[1] - x_end_offset]
        if person_label.sum() <= 0:
            print('A person label was <= zero (likely a person was labeled outside the image range).')
        label += person_label
    if force_full_image_count_normalize:
        label = head_count * (label / label.sum())
    return label


def make_gaussian(standard_deviation=1.0):
    """
    Make a square gaussian kernel.

    :param standard_deviation: The standard deviation of the 2D gaussian.
    :type standard_deviation: float | (float, float)
    :return: The gaussian array.
    :rtype: np.ndarray
    """
    try:
        x_standard_deviation = standard_deviation[0]
        y_standard_deviation = standard_deviation[1]
    except (IndexError, TypeError):
        x_standard_deviation = standard_deviation
        y_standard_deviation = standard_deviation
    x_off_center_size = int(x_standard_deviation * 2)
    y_off_center_size = int(y_standard_deviation * 2)
    x_linspace = np.linspace(-x_off_center_size, x_off_center_size, x_off_center_size * 2 + 1)
    y_linspace = np.linspace(-y_off_center_size, y_off_center_size, y_off_center_size * 2 + 1)
    x, y = np.meshgrid(x_linspace, y_linspace)
    x_part = (x ** 2) / (2.0 * x_standard_deviation ** 2)
    y_part = (y ** 2) / (2.0 * y_standard_deviation ** 2)
    gaussian_array = np.exp(-(x_part + y_part))
    return gaussian_array


def generate_point_density_map(head_positions, label_size):
    """Generates a density map where each persons density is concentrated the exact head position pixel."""
    density_map = np.zeros(label_size)
    out_of_bounds_count = 0
    for y, x in head_positions:
        try:
            y, x = int(round(y)), int(round(x))
            density_map[y, x] += 1
        except IndexError:
            out_of_bounds_count += 1
    return density_map, out_of_bounds_count


def generate_knn_map(head_positions, label_size, number_of_neighbors=1, upper_bound=None):
    """
    Generates a map of the nearest neighbor distances to head positions.

    :param head_positions: The list of head positions.
    :type head_positions: np.ndarray
    :param label_size: The size of the label.
    :type label_size: [int, int]
    :param number_of_neighbors: The number of neighbors to consider in the calculation.
    :type number_of_neighbors: int
    :param upper_bound: An upper bound on the allowed distance value.
    :type upper_bound: float
    :return: The map of the kNN distances.
    :rtype: np.ndarray
    """
    label_positions = permutations_of_shape_range(label_size)
    number_of_neighbors = min(number_of_neighbors, len(head_positions))
    nearest_neighbors_fit = NearestNeighbors(n_neighbors=number_of_neighbors,
                                             algorithm='ball_tree').fit(head_positions)
    neighbor_distances, _ = nearest_neighbors_fit.kneighbors(label_positions)
    if upper_bound is not None:
        neighbor_distances = np.clip(neighbor_distances, a_min=None, a_max=upper_bound)
    mean_neighbor_distances = neighbor_distances.mean(axis=1)
    knn_map = mean_neighbor_distances.reshape(label_size)
    return knn_map


def permutations_of_shape_range(shape):
    """
    Gives a flat array for all possible positions within an array based on the shape of the array.

    :param shape: The shape to get the positions for.
    :type shape: list[int]
    :return: All permutations of the shape positions in order.
    :rtype: np.ndarray
    """
    ranges = [np.arange(side) for side in shape]
    permutations = cartesian_product(ranges)
    return permutations


def cartesian_product(arrays):
    """
    Calculates the cartesian product of a list of arrays.

    :param arrays: The list of arrays.
    :type arrays: list[np.ndarray]
    :return: The cartesian product.
    :rtype: np.ndarray
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T