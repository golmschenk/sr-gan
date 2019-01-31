"""
Code for accessing the data in the database easily.
"""
import csv
import os
import shutil
import json

import patoolib
import requests
import torch
from urllib.request import urlretrieve
import tarfile
import numpy as np
import imageio
from skimage import transform, color
from torch.utils.data import Dataset
from scipy.io import loadmat
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader

from utility import to_normalized_range, download_and_extract_file


database_directory = '../Steering Angle Database'


class AgeDataset(Dataset):
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


class SteeringAngleDatabaseProcessor:
    """A class for preparing the Sully Chen steering angle database."""
    def __init__(self, preprocessed_image_size=128):
        self.preprocessed_image_size = preprocessed_image_size

    def download_and_preprocess(self):
        """Downloads and preprocesses the database."""
        print('Preparing steering angle database.')
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
        SteeringAngleDatabaseProcessor.download_file_from_google_drive('0B-KJCaaF7elleG1RbzVPZWV4Tlk',
                                                                       os.path.join(database_directory, 'temporary'))
        patoolib.extract_archive(os.path.join(database_directory, 'temporary'), outdir=database_directory)
        default_directory_name = 'driving_dataset'
        files = os.listdir(os.path.join(database_directory, default_directory_name))
        for file_ in files:
            shutil.move(os.path.join(database_directory, default_directory_name, file_), database_directory)
        shutil.rmtree(os.path.join(database_directory, default_directory_name))
        os.remove(os.path.join(database_directory, 'temporary'))

    @staticmethod
    def download_file_from_google_drive(id_, destination):
        """Google drive file downloader taken from here: https://stackoverflow.com/a/39225039/1191087"""
        def get_confirm_token(response_):
            for key, value in response_.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response_, destination_):
            CHUNK_SIZE = 32768
            with open(destination_, "wb") as f:
                for chunk in response_.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': id_}, stream=True)
        token = get_confirm_token(response)
        if token:
            params = {'id': id_, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        save_response_content(response, destination)

    @staticmethod
    def preprocess():
        """Preprocesses the database to the format needed by the network."""
        for file_name in os.listdir(database_directory):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(database_directory, file_name)
                image = imageio.imread(file_path).astype(np.uint8)
                np.save(file_path.replace('.jpg', '.npy'), image)
        meta_file_path = os.path.join(database_directory, 'data.txt')
        meta = np.genfromtxt(meta_file_path, delimiter=' ')
        angles = meta[:, 1]
        np.save(os.path.join(database_directory, 'angles.npy'), angles)


if __name__ == '__main__':
    SteeringAngleDatabaseProcessor().download()
    SteeringAngleDatabaseProcessor().preprocess()
