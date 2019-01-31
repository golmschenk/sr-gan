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


class SteeringAngleDataset(Dataset):
    """The dataset class for the age estimation application."""
    def __init__(self, dataset_path, start=None, end=None):
        self.dataset_path = database_directory
        meta = np.load(os.path.join(self.dataset_path, 'meta.npy'))
        image_names = meta[:, 0]
        angles = meta[:, 1]
        self.image_names = np.array(image_names[start:end])
        self.angles = np.array(angles[start:end], dtype=np.float32)
        self.length = self.angles.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = imageio.imread(os.path.join(self.dataset_path, image_name))
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image.astype(np.float32))
        image = to_normalized_range(image)
        angle = self.angles[index]
        angle = torch.tensor(angle, dtype=torch.float32)
        return image, angle


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
            if not file_.startswith('.'):
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

    def preprocess(self):
        """Preprocesses the database to the format needed by the network."""
        for file_name in os.listdir(database_directory):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(database_directory, file_name)
                image = imageio.imread(file_path).astype(np.uint8)
                image = transform.resize(image, (self.preprocessed_image_size, self.preprocessed_image_size),
                                         preserve_range=True)
                np.save(file_path.replace('.jpg', '.npy'), image)
        meta_file_path = os.path.join(database_directory, 'data.txt')
        meta = np.genfromtxt(meta_file_path, delimiter=' ')
        np.save(os.path.join(database_directory, 'meta.npy'), meta)


if __name__ == '__main__':
    SteeringAngleDatabaseProcessor().download()
    SteeringAngleDatabaseProcessor().preprocess()
