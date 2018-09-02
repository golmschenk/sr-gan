"""
Runs a batch of experiments.
"""
import torch

from age.sgan import AgeSganExperiment
from age.srgan import AgeExperiment
from coefficient.sgan import CoefficientSganExperiment
from coefficient.srgan import CoefficientExperiment
from crowd.sgan import CrowdSganExperiment
from crowd.srgan import CrowdExperiment
from settings import Settings, convert_to_settings_list
from utility import seed_all, clean_scientific_notation


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

application_name = 'crowd'
method_name = 'srgan'

settings_ = Settings()
if application_name == 'age':
    Experiment = AgeSganExperiment if method_name == 'sgan' else AgeExperiment
    settings_.unlabeled_loss_multiplier = [1e0]
    settings_.fake_loss_multiplier = [1e0]
    settings_.batch_size = 100
elif application_name == 'coef':
    Experiment = CoefficientSganExperiment if method_name == 'sgan' else CoefficientExperiment
    settings_.unlabeled_loss_multiplier = [1e-2]
    settings_.fake_loss_multiplier = [1e-2]
    settings_.batch_size = 1000
elif application_name == 'crowd':
    Experiment = CrowdSganExperiment if method_name == 'sgan' else CrowdExperiment
    settings_.unlabeled_loss_multiplier = [1e0]
    settings_.fake_loss_multiplier = [1e0]
    settings_.batch_size = 100
    settings_.number_of_cameras = [5]
    settings_.number_of_images_per_camera = [5]
else:
    raise ValueError('{} is not an available application.'.format(application_name))
settings_.unlabeled_dataset_size = [50000]
settings_.labeled_dataset_size = [1000]
settings_.summary_step_period = 1000
settings_.labeled_dataset_seed = [0]
settings_.steps_to_run = 100000
settings_.learning_rate = [1e-4]
settings_.gradient_penalty_multiplier = [0]
settings_.mean_offset = [0]
settings_.unlabeled_loss_order = 2
settings_.fake_loss_order = 0.5
settings_.generator_loss_order = 2
# settings_.load_model_path = '/home/golmschenk/srgan/logs/crowd 5c5i mostly trained'
settings_.local_setup()
settings_list = convert_to_settings_list(settings_)
seed_all(0)
for settings_ in settings_list:
    trial_name = 'spp test'
    trial_name += ' {}'.format(application_name)
    trial_name += ' {}'.format(method_name) if method_name != 'srgan' else ''
    if application_name == 'crowd':
        trial_name += ' c{}i{}'.format(settings_.number_of_cameras, settings_.number_of_images_per_camera)
    else:
        trial_name += ' le{}'.format(settings_.labeled_dataset_size)
        trial_name += ' ue{}'.format(settings_.unlabeled_dataset_size)
    trial_name += ' ul{:e}'.format(settings_.unlabeled_loss_multiplier)
    trial_name += ' fl{:e}'.format(settings_.fake_loss_multiplier)
    trial_name += ' gp{:e}'.format(settings_.gradient_penalty_multiplier)
    trial_name += ' mo{:e}'.format(settings_.mean_offset)
    trial_name += ' lr{:e}'.format(settings_.learning_rate)
    trial_name += ' gs{}'.format(settings_.generator_training_step_period)
    trial_name += ' ls{}'.format(settings_.labeled_dataset_seed)
    trial_name += ' u{}f{}g{}'.format(settings_.unlabeled_loss_order,
                                      settings_.fake_loss_order,
                                      settings_.generator_loss_order)
    trial_name += ' bs{}'.format(settings_.batch_size)
    trial_name += ' nf' if settings_.normalize_fake_loss else ''
    trial_name += ' l' if settings_.load_model_path else ''
    settings_.trial_name = clean_scientific_notation(trial_name)
    experiment = Experiment(settings_)
    experiment.train()
    if experiment.signal_quit:
        break
