"""
Code for the abstract base application class.
"""
from typing import Tuple
from abc import ABC, abstractmethod
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

class Application(ABC):
    @abstractmethod
    def dataset_setup(self, experiment: 'Experiment') -> Tuple[Dataset, DataLoader, Dataset, DataLoader, Dataset]:
        train_dataset = Dataset()
        validation_dataset = Dataset()
        test_dataset = Dataset()
        train_dataset_loader = DataLoader(train_dataset)
        validation_dataset_loader = DataLoader(validation_dataset)
        return train_dataset, train_dataset_loader, validation_dataset, validation_dataset_loader, test_dataset

    @abstractmethod
    def model_setup(self) -> Tuple[Module, Module, Module]:
        dnn_model = Module()
        d_model = Module()
        g_model = Module()
        return dnn_model, d_model, g_model

    @abstractmethod
    def validation_summaries(self, experiment: 'Experiment', step: int):
        pass
