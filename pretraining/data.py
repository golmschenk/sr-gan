"""
Code for the ImageNet database.
"""
import os
import json
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset


from utility import to_normalized_range


class ImageNetDataset(Dataset):
    """The dataset class for the age estimation application."""
    def __init__(self, dataset_path, start=None, end=None, gender_filter=None):
        if gender_filter is not None:
            raise NotImplementedError()
        self.dataset_path = dataset_path
        with open(os.path.join(self.dataset_path, 'meta.json')) as json_file:
            json_contents = json.load(json_file)
        image_names, ages = [], []
        for entry in json_contents:
            if isinstance(entry, dict):
                image_names.append(entry['image_name'])
                ages.append(entry['age'])
            else:
                image_name, age, gender = entry
                image_names.append(image_name)
                ages.append(age)
        self.image_names = np.array(image_names[start:end])
        self.ages = np.array(ages[start:end], dtype=np.float32)
        self.length = self.ages.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = imageio.imread(os.path.join(self.dataset_path, image_name))
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image.astype(np.float32))
        image = to_normalized_range(image)
        age = self.ages[idx]
        age = torch.tensor(age, dtype=torch.float32)
        return image, age

