"""
Code for generating labels from head positions.
"""
import numpy as np


head_standard_deviation_meters = 0.2
body_width_standard_deviation_meters = 0.2
body_height_standard_deviation_meters = 0.5
body_height_offset_meters = 0.875

default_head_standard_deviation = 20

head_count = 0


def generate_density_label(head_positions, label_size, perspective=None, include_body=True, ignore_tiny=False,
                           force_full_image_count_normalize=True):
    """
    Generates a density label given the head positions and other meta data.

    :param head_positions: The head labeling positions.
    :type head_positions: np.ndarray
    :param perspective: The perspective map.
    :type perspective: np.ndarray
    :return: The density labeling.
    :rtype: np.ndarray
    """
    global head_count
    body_parts = 2 if include_body else 1
    label = np.zeros(shape=label_size, dtype=np.float32)
    for head_position in head_positions:
        x, y = head_position.astype(np.uint32)
        if perspective is not None:
            if 0 <= x < perspective.shape[1]:
                position_perspective = perspective[y, x]
            else:
                position_perspective = perspective[y, 0]
            if ignore_tiny and position_perspective < 3.1:
                continue
            head_standard_deviation = position_perspective * head_standard_deviation_meters
        else:
            head_standard_deviation = default_head_standard_deviation
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
            raise ValueError('Person label should not be less than zero (though it\'s possible if the person was labeled outside the image).')
        label += person_label
    if force_full_image_count_normalize:
        label = label / label.sum()
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
