"""
General settings.
"""
import platform
import random
from copy import deepcopy


class Settings:
    """Represents the settings for a given run of SRGAN."""
    def __init__(self):
        self.trial_name = 'base'
        self.application = 'coefficient'
        self.steps_to_run = 200000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 1000
        self.summary_step_period = 1000
        self.labeled_dataset_size = 50
        self.unlabeled_dataset_size = 50000
        self.validation_dataset_size = 1000
        self.learning_rate = 1e-5

        self.labeled_loss_multiplier = 1e0
        self.unlabeled_loss_multiplier = 1e0
        self.fake_loss_multiplier = 1e0
        self.gradient_penalty_on = True
        self.gradient_penalty_multiplier = 1e1
        self.norm_loss_multiplier = 1
        self.noise_scale = 5e-1
        self.mean_offset = 1e0
        self.fake_loss_order = 2
        self.unlabeled_loss_order = 2
        self.generator_loss_order = 2
        self.generator_training_step_period = 1
        self.labeled_dataset_seed = 0

        self.load_model_path = None
        self.should_save_models = False
        self.skip_completed_experiment = True

    def local_setup(self):
        """Code to override some settings when debugging on the local (low power) machine."""
        if 'Carbon' in platform.node():
            self.labeled_dataset_seed = 0
            self.batch_size = 10
            self.summary_step_period = 10
            self.labeled_dataset_size = 10
            self.unlabeled_dataset_size = 10
            self.validation_dataset_size = 10


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
