"""
Code from preprocessing the UCSD dataset.
"""
import os
import random
import imageio
import numpy as np
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NegativeOneToOneNormalizeImage, NumpyArraysToTorchTensors
from crowd.database_preprocessor import DatabasePreprocessor

from utility import seed_all


class UcfQnrfFullImageDataset(Dataset):
    """
    A class for the UCF QNRF full image crowd dataset.
    """
    def __init__(self, dataset='train', seed=None, number_of_examples=None, map_directory_name='maps',
                 examples_start=None):
        seed_all(seed)
        if examples_start is None:
            examples_end = number_of_examples
        elif number_of_examples is None:
            examples_end = None
        else:
            examples_end = examples_start + number_of_examples
        seed_all(seed)
        self.dataset_directory = os.path.join(UcfQnrfPreprocessor().database_directory, dataset.capitalize())
        file_names = os.listdir(os.path.join(self.dataset_directory, 'labels'))
        random.shuffle(file_names)
        self.file_names = [name for name in file_names if name.endswith('.npy')][examples_start:examples_end]
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


class UcfQnrfTransformedDataset(Dataset):
    """
    A class for the transformed UCF QNRF crowd dataset.
    """
    def __init__(self, dataset='train', image_patch_size=224, label_patch_size=224, seed=None, number_of_examples=None,
                 middle_transform=None, map_directory_name='maps', examples_start=None):
        seed_all(seed)
        if examples_start is None:
            examples_end = number_of_examples
        elif number_of_examples is None:
            examples_end = None
        else:
            examples_end = examples_start + number_of_examples
        self.dataset_directory = os.path.join(UcfQnrfPreprocessor().database_directory, dataset.capitalize())
        file_names = os.listdir(os.path.join(self.dataset_directory, 'labels'))
        random.shuffle(file_names)
        self.file_names = [name for name in file_names if name.endswith('.npy')][examples_start:examples_end]
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


class UcfQnrfPreprocessor(DatabasePreprocessor):
    """The preprocessor for the ShanghaiTech dataset."""
    def __init__(self):
        super().__init__()
        self.database_name = 'UCF QNRF'
        self.database_url = 'http://crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip'
        self.database_archived_directory_name = 'UCF-QNRF_ECCV18'

    def preprocess(self):
        """Preprocesses the database generating the image and map labels."""
        for dataset_name_ in ['Train', 'Test']:
            dataset_directory = os.path.join(self.database_directory, dataset_name_)
            for mat_filename in os.listdir(os.path.join(self.database_directory, dataset_name_)):
                if not mat_filename.endswith('.mat'):
                    continue
                file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
                mat_path = os.path.join(self.database_directory, dataset_name_, mat_filename)
                original_image_path = os.path.join(self.database_directory, dataset_name_, file_name + '.jpg')
                image = imageio.imread(original_image_path)
                mat = scipy.io.loadmat(mat_path)
                original_head_positions = mat['annPoints']  # x, y ordering (mostly).
                # Get y, x ordering.
                head_positions = self.get_y_x_head_positions(original_head_positions, file_name,
                                                             label_size=image.shape[:2])
                self.generate_labels_for_example(dataset_directory, file_name, image, head_positions)

    @staticmethod
    def get_y_x_head_positions(original_head_positions, file_name, label_size):
        """Swaps the x's and y's of the head positions. Accounts for files where the labeling is incorrect."""
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


if __name__ == '__main__':
    preprocessor = UcfQnrfPreprocessor()
    preprocessor.download_and_preprocess()
