"""
Runs a batch of experiments.
"""
import torch

from age.sgan import AgeSganExperiment
from age.srgan import AgeExperiment
from coefficient.dggan import CoefficientDgganExperiment
from coefficient.sgan import CoefficientSganExperiment
from coefficient.srgan import CoefficientExperiment
from crowd.data import CrowdDataset
from crowd.dggan import CrowdDgganExperiment
from crowd.dnn import CrowdDnnExperiment
from crowd.sgan import CrowdSganExperiment
from crowd.srgan import CrowdExperiment
from driving.srgan import DrivingExperiment
from settings import Settings, convert_to_settings_list, ApplicationName, MethodName
from utility import seed_all, clean_scientific_notation, abs_plus_one_sqrt_mean_neg, square_mean, abs_mean_neg, \
    abs_plus_one_log_neg, abs_plus_one_log_mean_neg, norm_mean

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

application_name = ApplicationName.crowd
method_name = MethodName.dnn

settings_ = Settings()
if application_name == ApplicationName.age:
    Experiment = {MethodName.srgan: AgeExperiment, MethodName.sgan: AgeSganExperiment}[method_name]
    settings_.unlabeled_loss_multiplier = [1e2]
    settings_.fake_loss_multiplier = [1e1]
    settings_.batch_size = 600
    settings_.unlabeled_dataset_size = 50000
    settings_.labeled_dataset_size = [5000]
    settings_.gradient_penalty_multiplier = 1e2
elif application_name == ApplicationName.driving:
    Experiment = DrivingExperiment
    settings_.unlabeled_loss_multiplier = [1e2]
    settings_.fake_loss_multiplier = [1e1]
    settings_.batch_size = 600
    settings_.unlabeled_dataset_size = None
    settings_.labeled_dataset_size = [7200, 4000]
    settings_.validation_dataset_size = 9000
    settings_.gradient_penalty_multiplier = 1e2
elif application_name == ApplicationName.coefficient:
    Experiment = {MethodName.srgan: CoefficientExperiment, MethodName.sgan: CoefficientSganExperiment,
                  MethodName.dggan: CoefficientDgganExperiment}[method_name]
    settings_.unlabeled_loss_multiplier = [1e-1, 1e0, 1e1]
    settings_.fake_loss_multiplier = [1e-1, 1e0, 1e1]
    settings_.batch_size = 5000
    settings_.unlabeled_dataset_size = 50000
    settings_.labeled_dataset_size = [500]
    settings_.gradient_penalty_multiplier = 1e1
elif application_name == ApplicationName.crowd:
    Experiment = {MethodName.srgan: CrowdExperiment, MethodName.sgan: CrowdSganExperiment,
                  MethodName.dnn: CrowdDnnExperiment, MethodName.dggan: CrowdDgganExperiment}[method_name]
    settings_.unlabeled_loss_multiplier = [1e3]
    settings_.fake_loss_multiplier = [1e2]
    settings_.batch_size = 40
    settings_.number_of_cameras = [5]
    settings_.number_of_images_per_camera = [5]
    settings_.crowd_dataset = CrowdDataset.ucf_qnrf
    settings_.labeled_loss_order = 2
    settings_.unlabeled_dataset_size = None
    settings_.labeled_dataset_size = None
    settings_.gradient_penalty_multiplier = 1e2
    settings_.map_directory_name = ['i1nn_maps', 'i2nn_maps', 'i3nn_maps', 'i4nn_maps', 'i5nn_maps', 'i6nn_maps']
    settings_.map_multiplier = 1e-6
else:
    raise ValueError('{} is not an available application.'.format(application_name))
settings_.summary_step_period = 1000
settings_.labeled_dataset_seed = 0
settings_.steps_to_run = 200000
settings_.learning_rate = [1e-4]
settings_.map_multiplier = 1
# settings_.load_model_path = 'logs/k comparison i1nn_maps ShanghaiTech crowd dnn ul1e3 fl1e2 gp1e2 lr1e-4 mm1e-6 ls0 bs40'
settings_.contrasting_distance_function = abs_plus_one_sqrt_mean_neg
settings_.matching_distance_function = square_mean
settings_.continue_existing_experiments = True
settings_.local_setup()
settings_list = convert_to_settings_list(settings_, shuffle=False)
seed_all(0)
previous_trial_directory = None
for settings_ in settings_list:
    trial_name = 'k comparison'
    trial_name += ' {}'.format(settings_.map_directory_name) if application_name == ApplicationName.crowd else ''
    trial_name += f' {settings_.crowd_dataset.value}' if application_name == ApplicationName.crowd else ''
    trial_name += ' {}'.format(application_name.value)
    trial_name += ' {}'.format(method_name.value) if method_name != MethodName.srgan else ''
    if method_name != MethodName.dnn:
        if application_name == ApplicationName.crowd and settings_.crowd_dataset == CrowdDataset.world_expo:
            trial_name += ' c{}i{}'.format(settings_.number_of_cameras, settings_.number_of_images_per_camera)
        else:
            trial_name += ' le{}'.format(settings_.labeled_dataset_size)
            trial_name += ' ue{}'.format(settings_.unlabeled_dataset_size)
    trial_name += ' ul{:e}'.format(settings_.unlabeled_loss_multiplier)
    trial_name += ' fl{:e}'.format(settings_.fake_loss_multiplier)
    trial_name += ' gp{:e}'.format(settings_.gradient_penalty_multiplier)
    trial_name += ' lr{:e}'.format(settings_.learning_rate)
    trial_name += ' mm{:e}'.format(settings_.map_multiplier) if application_name == ApplicationName.crowd else ''
    trial_name += ' ls{}'.format(settings_.labeled_dataset_seed)
    trial_name += ' bs{}'.format(settings_.batch_size)
    trial_name += ' l' if settings_.load_model_path else ''
    settings_.trial_name = clean_scientific_notation(trial_name)
    if previous_trial_directory and settings_.continue_from_previous_trial:
        settings_.load_model_path = previous_trial_directory
    experiment = Experiment(settings_)
    experiment.train()
    previous_trial_directory = experiment.trial_directory
    if experiment.signal_quit:
        break
