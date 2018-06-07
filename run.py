"""
Runs a batch of experiments.
"""
import torch

from age_application import AgeApplication
from coefficient_application import CoefficientApplication
from crowd_application import CrowdApplication
from settings import Settings, convert_to_settings_list
from srgan import Experiment
from utility import seed_all, clean_scientific_notation


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


settings_ = Settings()
settings_.application = CrowdApplication()
settings_.unlabeled_dataset_size = [50000]
settings_.batch_size = 100
settings_.summary_step_period = 1000
settings_.labeled_dataset_seed = [0, 1, 2]
settings_.labeled_dataset_size = [100]
settings_.unlabeled_loss_multiplier = [1e-2]
settings_.fake_loss_multiplier = [1e-2]
settings_.steps_to_run = 1000000
settings_.learning_rate = [1e-4]
settings_.gradient_penalty_multiplier = [1e1]
settings_.mean_offset = [0]
settings_.unlabeled_loss_order = 2
settings_.fake_loss_order = [0.5]
settings_.generator_loss_order = [2]
#settings_.load_model_path = '/home/golmschenk/srgan/logs/age ul1e0 fl1e0 le3000 gp1e1 bg2e0 lr1e-5 nl0 gs1 ls0 u2f0.5g2 y2018m04d20h22m58s03'
settings_.local_setup()
settings_list = convert_to_settings_list(settings_)
seed_all(0)
for settings_ in settings_list:
    trial_name = 'coef gp0'
    trial_name += ' ul{:e}'.format(settings_.unlabeled_loss_multiplier)
    trial_name += ' fl{:e}'.format(settings_.fake_loss_multiplier)
    trial_name += ' le{}'.format(settings_.labeled_dataset_size)
    trial_name += ' gp{:e}'.format(settings_.gradient_penalty_multiplier)
    trial_name += ' mo{:e}'.format(settings_.mean_offset)
    trial_name += ' lr{:e}'.format(settings_.learning_rate)
    trial_name += ' nl{}'.format(settings_.norm_loss_multiplier)
    trial_name += ' gs{}'.format(settings_.generator_training_step_period)
    trial_name += ' ls{}'.format(settings_.labeled_dataset_seed)
    trial_name += ' u{}f{}g{}'.format(settings_.unlabeled_loss_order,
                                      settings_.fake_loss_order,
                                      settings_.generator_loss_order)
    trial_name += ' ue{}'.format(settings_.unlabeled_dataset_size)
    trial_name += ' ns' if not settings_.normalize_fake_loss else ''
    trial_name += ' l' if settings_.load_model_path else ''
    settings_.trial_name = clean_scientific_notation(trial_name)
    experiment = Experiment(settings_)
    experiment.train()
    if experiment.signal_quit:
        break
