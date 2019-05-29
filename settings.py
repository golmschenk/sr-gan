"""
General settings.
"""
import platform
import random
from copy import deepcopy
from enum import Enum

from utility import abs_plus_one_sqrt_mean_neg, abs_mean


class Settings:
    """Represents the settings for a given run of SRGAN."""
    def __init__(self):
        self.trial_name = 'base'
        self.steps_to_run = 200000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 1000
        self.summary_step_period = 2000
        self.labeled_dataset_size = 50
        self.unlabeled_dataset_size = 50000
        self.validation_dataset_size = 1000
        self.learning_rate = 1e-4
        self.weight_decay = 0

        self.labeled_loss_multiplier = 1e0
        self.matching_loss_multiplier = 1e0
        self.contrasting_loss_multiplier = 1e0
        self.srgan_loss_multiplier = 1e0
        self.dggan_loss_multiplier = 1e1
        self.gradient_penalty_on = True
        self.gradient_penalty_multiplier = 1e1
        self.mean_offset = 0
        self.labeled_loss_order = 2
        self.generator_training_step_period = 1
        self.labeled_dataset_seed = 0
        self.normalize_fake_loss = False
        self.normalize_feature_norm = False
        self.contrasting_distance_function = abs_plus_one_sqrt_mean_neg
        self.matching_distance_function = abs_mean

        self.load_model_path = None
        self.should_save_models = True
        self.skip_completed_experiment = True
        self.number_of_data_workers = 4
        self.pin_memory = True
        self.continue_from_previous_trial = False
        self.continue_existing_experiments = False
        self.save_step_period = None

        # Coefficient application only.
        self.hidden_size = 10

        # Crowd application only.
        self.crowd_dataset = 'World Expo'
        self.number_of_cameras = 5  # World Expo data only
        self.number_of_images_per_camera = 5  # World Expo data only
        self.test_summary_size = None
        self.test_sliding_window_size = 128
        self.image_patch_size = 224
        self.label_patch_size = 224
        self.map_multiplier = 1e-6
        self.map_directory_name = 'i1nn_maps'

        # SGAN models only.
        self.number_of_bins = 10

    def local_setup(self):
        """Code to override some settings when debugging on the local (low power) machine."""
        if 'Carbon' in platform.node():
            self.labeled_dataset_seed = 0
            self.batch_size = min(10, self.batch_size)
            self.summary_step_period = 10
            self.labeled_dataset_size = 10
            self.unlabeled_dataset_size = 10
            self.validation_dataset_size = 10
            self.skip_completed_experiment = False
            self.number_of_data_workers = 0


def convert_to_settings_list(settings, shuffle=True):
    """
    Creates permutations of settings for any setting that is a list.
    (e.g. if `learning_rate = [1e-4, 1e-5]` and `batch_size = [10, 100]`, a list of 4 settings objects will return)
    This function is black magic. Beware.
    """
    settings_list = [settings]
    next_settings_list = []
    any_contains_list = True
    while any_contains_list:
        any_contains_list = False
        for settings in settings_list:
            contains_list = False
            for attribute_name, attribute_value in vars(settings).items():
                if isinstance(attribute_value, (list, tuple)):
                    for value in attribute_value:
                        settings_copy = deepcopy(settings)
                        setattr(settings_copy, attribute_name, value)
                        next_settings_list.append(settings_copy)
                    contains_list = True
                    any_contains_list = True
                    break
            if not contains_list:
                next_settings_list.append(settings)
        settings_list = next_settings_list
        next_settings_list = []
    if shuffle:
        random.seed()
        random.shuffle(settings_list)
    return settings_list


class ApplicationName(Enum):
    """An enum to select the application of the code to run."""
    coefficient = 'coefficient'
    age = 'age'
    crowd = 'crowd'
    driving = 'driving'


class MethodName(Enum):
    """An enum to select the method of the code to run."""
    srgan = 'srgan'
    dnn = 'dnn'
    dggan = 'dggan'
    sgan = 'sgan'
