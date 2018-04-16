"""
General settings.
"""
import random
from copy import deepcopy


class Settings():
    def __init__(self):
        self.trial_name = 'base'
        self.steps_to_run = 200000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 1000
        self.presentation_step_period = 1000
        self.summary_step_period = 1000
        self.labeled_dataset_size = 50
        self.unlabeled_dataset_size = 50000
        self.test_dataset_size = 1000
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


def convert_to_settings_list(settings, shuffle=True):
    """
    Creates permutations of settings for any setting that is a list.
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
        random.shuffle(settings_list)
    return settings_list
