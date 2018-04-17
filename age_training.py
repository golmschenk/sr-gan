from torch.utils.data import DataLoader

from age_data import AgeDataset
from utility import seed_all


def dataset_setup(settings):
    seed_all(settings.labeled_dataset_seed)
    train_dataset = AgeDataset(start=0, end=settings.labeled_dataset_size)
    train_dataset_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    unlabeled_dataset = AgeDataset(start=train_dataset.length,
                                   end=train_dataset.length + settings.unlabeled_dataset_size)
    unlabeled_dataset_loader = DataLoader(unlabeled_dataset, batch_size=settings.batch_size, shuffle=True)
    train_unlabeled_dataset_size = train_dataset.length + unlabeled_dataset.length
    validation_dataset = AgeDataset(start=train_unlabeled_dataset_size,
                                    end=train_unlabeled_dataset_size + settings.validation_dataset_size)
    return train_dataset, train_dataset_loader, unlabeled_dataset, unlabeled_dataset_loader, validation_dataset


def model_setup():
    return None