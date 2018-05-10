"""
Regression semi-supervised GAN code.
"""
import datetime
import os
import select
import sys

from torch.nn import Module
from torch.optim import Adam
import torch
from torch.utils.data import Dataset, DataLoader

from coefficient_application import CoefficientApplication
from age_application import AgeApplication
from settings import Settings, convert_to_settings_list
from training_functions import dnn_training_step, gan_training_step
from utility import SummaryWriter, infinite_iter, clean_scientific_notation, gpu, seed_all, make_directory_name_unique


should_quit = False


class Experiment:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.trial_directory: str = None
        self.dnn_summary_writer: SummaryWriter = None
        self.gan_summary_writer: SummaryWriter = None
        self.train_dataset: Dataset = None
        self.train_dataset_loader: DataLoader = None
        self.unlabeled_dataset: Dataset = None
        self.unlabeled_dataset_loader: DataLoader = None
        self.validation_dataset: Dataset = None
        self.DNN: Module = None
        self.D: Module = None
        self.G: Module = None

    def train(self):
        """
        Run the SRGAN training for the experiment.
        """
        self.trial_directory = os.path.join(self.settings.logs_directory, self.settings.trial_name)
        if self.settings.skip_completed_experiment and os.path.exists(self.trial_directory) and '/check' not in self.trial_directory:
            print('{} experiment already exists. Skipping...'.format(self.trial_directory))
            return
        self.trial_directory = make_directory_name_unique(self.trial_directory)
        print(self.trial_directory)
        os.makedirs(os.path.join(self.trial_directory, self.settings.temporary_directory))
        self.dnn_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'DNN'))
        self.gan_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'GAN'))
        self.dnn_summary_writer.summary_period = self.settings.summary_step_period
        self.gan_summary_writer.summary_period = self.settings.summary_step_period


        dataset_setup = self.settings.application.dataset_setup
        model_setup = self.settings.application.model_setup
        validation_summaries = self.settings.application.validation_summaries

        self.train_dataset, self.train_dataset_loader, self.unlabeled_dataset, self.unlabeled_dataset_loader, self.validation_dataset = dataset_setup(self)
        DNN_model, D_model, G_model = model_setup()

        if self.settings.load_model_path:
            if not torch.cuda.is_available():
                map_location = 'cpu'
            else:
                map_location = None
            DNN_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'DNN_model.pth'), map_location))
            D_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'D_model.pth'), map_location))
            G_model.load_state_dict(torch.load(os.path.join(self.settings.load_model_path, 'G_model.pth'), map_location))
        self.G = G_model.to(gpu)
        self.D = D_model.to(gpu)
        self.DNN = DNN_model.to(gpu)
        d_lr = self.settings.learning_rate
        g_lr = d_lr

        betas = (0.9, 0.999)
        weight_decay = 1e-2
        D_optimizer = Adam(self.D.parameters(), lr=d_lr, weight_decay=weight_decay)
        G_optimizer = Adam(self.G.parameters(), lr=g_lr)
        DNN_optimizer = Adam(self.DNN.parameters(), lr=d_lr, weight_decay=weight_decay)

        step_time_start = datetime.datetime.now()
        train_dataset_generator = infinite_iter(self.train_dataset_loader)
        unlabeled_dataset_generator = infinite_iter(self.unlabeled_dataset_loader)

        for step in range(self.settings.steps_to_run):
            # DNN.
            labeled_examples, labels = next(train_dataset_generator)
            labeled_examples, labels = labeled_examples.to(gpu), labels.to(gpu)
            dnn_training_step(self.DNN, DNN_optimizer, self.dnn_summary_writer, labeled_examples, labels, self.settings, step)
            # GAN.
            unlabeled_examples, _ = next(unlabeled_dataset_generator)
            unlabeled_examples = unlabeled_examples.to(gpu)
            gan_training_step(self.D, D_optimizer, self.G, G_optimizer, self.gan_summary_writer, labeled_examples, labels, self.settings, step,
                              unlabeled_examples)

            if self.gan_summary_writer.is_summary_step():
                print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                step_time_start = datetime.datetime.now()

                self.D.eval()
                self.DNN.eval()
                self.G.eval()
                validation_summaries(self, step)
                self.D.train()
                self.DNN.train()
                self.G.train()
                while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if 'save' in line:
                        torch.save(self.DNN.state_dict(), os.path.join(self.trial_directory, 'DNN_model_{}.pth'.format(step)))
                        torch.save(self.D.state_dict(), os.path.join(self.trial_directory, 'D_model_{}.pth'.format(step)))
                        torch.save(self.G.state_dict(), os.path.join(self.trial_directory, 'G_model_{}.pth'.format(step)))
                        print('\rSaved model for step {}...'.format(step))
                    if 'quit' in line:
                        global should_quit
                        should_quit = True

        print('Completed {}'.format(self.trial_directory))
        if self.settings.should_save_models:
            torch.save(self.DNN.state_dict(), os.path.join(self.trial_directory, 'DNN_model.pth'))
            torch.save(self.D.state_dict(), os.path.join(self.trial_directory, 'D_model.pth'))
            torch.save(self.G.state_dict(), os.path.join(self.trial_directory, 'G_model.pth'))


if __name__ == '__main__':
    settings_ = Settings()
    settings_.application = AgeApplication()
    settings_.unlabeled_dataset_size = [50000]
    settings_.batch_size = 50
    settings_.summary_step_period = 1000
    settings_.labeled_dataset_seed = [0]
    settings_.labeled_dataset_size = [1000]
    settings_.unlabeled_loss_multiplier = [1e0]
    settings_.fake_loss_multiplier = [1e0]
    settings_.steps_to_run = 150000
    settings_.learning_rate = [1e-4]
    settings_.gradient_penalty_multiplier = [1e1]
    settings_.norm_loss_multiplier = [0]
    settings_.mean_offset = 2
    settings_.unlabeled_loss_order = 2
    settings_.fake_loss_order = [0.5]
    settings_.generator_loss_order = 2
    settings_.generator_training_step_period = 1
    settings_.should_save_models = True
    #settings_.load_model_path = '/home/golmschenk/srgan/logs/age ul1e0 fl1e0 le3000 gp1e1 bg2e0 lr1e-5 nl0 gs1 ls0 u2f0.5g2 y2018m04d20h22m58s03'
    settings_.local_setup()
    settings_list = convert_to_settings_list(settings_)
    seed_all(0)
    for settings_ in settings_list:
        trial_name = 'check'
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
        trial_name += ' l' if settings_.load_model_path else ''
        settings_.trial_name = clean_scientific_notation(trial_name)
        experiment = Experiment(settings_)
        experiment.train()
        if should_quit:
            break
