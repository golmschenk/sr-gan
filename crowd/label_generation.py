"""
Code for generating labels from head positions.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


head_standard_deviation_meters = 0.2
body_width_standard_deviation_meters = 0.2
body_height_standard_deviation_meters = 0.5
body_height_offset_meters = 0.875

dataset_head_count = 0


def generate_density_label(head_positions, label_size, perspective=None, include_body=True, ignore_tiny=False,
                           force_full_image_count_normalize=True, perspective_resizing=True):
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
    :return: The density labeling.
    :rtype: np.ndarray
    """
    global dataset_head_count
    head_count = 0
    body_parts = 2 if include_body else 1
    number_of_neighbors = min(11, len(head_positions))
    nearest_neighbors = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='ball_tree').fit(head_positions)
    neighbor_distances, _ = nearest_neighbors.kneighbors(head_positions)
    average_neighbor_distances = neighbor_distances[0:].mean(axis=1)
    label = np.zeros(shape=label_size, dtype=np.float32)
    for head_index, head_position in enumerate(head_positions):
        x, y = head_position.astype(np.uint32)
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
                neighbor_deviation_beta = 0.15
                head_standard_deviation = average_neighbor_distances[head_index] * neighbor_deviation_beta
                position_perspective = None
        else:
            head_standard_deviation = 8
            position_perspective = None
        head_gaussian = make_gaussian(head_standard_deviation)
        head_gaussian = head_gaussian / (body_parts * head_gaussian.sum())
        dataset_head_count += 1
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
        person_label[y - off_center_size + y_start_offset:y + off_center_size + 1 - y_end_offset,
                     x - off_center_size + x_start_offset:x + off_center_size + 1 - x_end_offset
                     ] += head_gaussian[y_start_offset:head_gaussian.shape[0] - y_end_offset,
                                        x_start_offset:head_gaussian.shape[1] - x_end_offset]
        if perspective is not None or not include_body:
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
