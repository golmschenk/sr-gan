"""
Code for accessing the data in the database easily.
"""
import math
import os
import shutil
import patoolib
import requests
import torch
import numpy as np
import pandas as pd
import imageio
from skimage import transform
import sklearn.utils
from torch.utils.data import Dataset

from utility import to_normalized_range, seed_all

database_directory = '../Steering Angle Database'


class SteeringAngleDataset(Dataset):
    """The dataset class for the age estimation application."""
    def __init__(self, start=None, end=None, seed=None, batch_size=None):
        seed_all(seed)
        self.dataset_path = database_directory
        meta = pd.read_pickle(os.path.join(self.dataset_path, 'meta.pkl'))
        meta = sklearn.utils.shuffle(meta, random_state=seed)  # Shuffles only first axis.
        image_names = meta.iloc[:, 0].values
        angles = meta.iloc[:, 1].values
        self.image_names = np.array(image_names[start:end])
        self.angles = np.array(angles[start:end], dtype=np.float32)
        # Force full batch sizes
        if self.image_names.shape[0] < batch_size:
            repeats = math.ceil(batch_size / self.image_names.shape[0])
            self.image_names = np.repeat(self.image_names, repeats)
            self.angles = np.repeat(self.angles, repeats)
        self.length = self.angles.shape[0]
        self.image_size = 128

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = np.load(os.path.join(self.dataset_path, image_name.replace('.jpg', '.npy')))
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
        # noinspection SpellCheckingInspection
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

        # noinspection PyMissingOrEmptyDocstring
        def get_confirm_token(response_):
            for key, value in response_.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        # noinspection PyMissingOrEmptyDocstring
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
        meta_file_path = os.path.join(database_directory, 'data.txt')
        meta = pd.read_csv(meta_file_path, delimiter=' ', header=None)
        meta = meta[meta[0] != '45567.jpg']  # Corrupt image.
        meta.to_pickle(os.path.join(database_directory, 'meta.pkl'))
        for file_name in meta.iloc[:, 0].values:
            if file_name.endswith('.jpg'):
                file_path = os.path.join(database_directory, file_name)
                image = imageio.imread(file_path).astype(np.uint8)
                image = transform.resize(image, (self.preprocessed_image_size, self.preprocessed_image_size),
                                         preserve_range=True)
                image = image.transpose((2, 0, 1))
                np.save(file_path.replace('.jpg', '.npy'), image)


if __name__ == '__main__':
    SteeringAngleDatabaseProcessor().download_and_preprocess()
