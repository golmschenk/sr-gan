"""
Code to run only the DNN model.
"""
import datetime
import os
import re
from abc import ABC
import torch
from torch.optim import Adam

from srgan import Experiment
from utility import SummaryWriter, gpu


class DnnExperiment(Experiment, ABC):
    """A class to manage an experimental trial with only a DNN."""
    def prepare_summary_writers(self):
        """Prepares the summary writers for TensorBoard."""
        self.dnn_summary_writer = SummaryWriter(os.path.join(self.trial_directory, 'DNN'))
        self.dnn_summary_writer.summary_period = self.settings.summary_step_period

    def gpu_mode(self):
        """
        Moves the networks to the GPU (if available).
        """
        self.DNN.to(gpu)

    def eval_mode(self):
        """
        Changes the network to evaluation mode.
        """
        self.DNN.eval()

    def train_mode(self):
        """
        Converts the networks to train mode.
        """
        self.DNN.train()

    def prepare_optimizers(self):
        """Prepares the optimizers of the network."""
        d_lr = self.settings.learning_rate
        weight_decay = self.settings.weight_decay
        self.dnn_optimizer = Adam(self.DNN.parameters(), lr=d_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    def save_models(self, step):
        """Saves the network models."""
        model = {'DNN': self.DNN.state_dict(),
                 'dnn_optimizer': self.dnn_optimizer.state_dict(),
                 'step': step}
        torch.save(model, os.path.join(self.trial_directory, f'model_{step}.pth'))

    def load_models(self, with_optimizers=True):
        """Loads existing models if they exist at `self.settings.load_model_path`."""
        if self.settings.load_model_path:
            latest_model = None
            model_path_file_names = os.listdir(self.settings.load_model_path)
            for file_name in model_path_file_names:
                match = re.search(r'model_?(\d+)?\.pth', file_name)
                if match:
                    latest_model = self.compare_model_path_for_latest(latest_model, match)
            latest_model = None if latest_model is None else latest_model.group(0)
            if not torch.cuda.is_available():
                map_location = 'cpu'
            else:
                map_location = None
            if latest_model:
                model_path = os.path.join(self.settings.load_model_path, latest_model)
                loaded_model = torch.load(model_path, map_location)
                self.DNN.load_state_dict(loaded_model['DNN'])
                if with_optimizers:
                    self.dnn_optimizer.load_state_dict(loaded_model['dnn_optimizer'])
                    self.optimizer_to_gpu(self.dnn_optimizer)
                print('Model loaded from `{}`.'.format(model_path))
                if self.settings.continue_existing_experiments:
                    self.starting_step = loaded_model['step'] + 1
                    print(f'Continuing from step {self.starting_step}')

    def training_loop(self):
        """Runs the main training loop."""
        train_dataset_generator = self.infinite_iter(self.train_dataset_loader)
        step_time_start = datetime.datetime.now()
        for step in range(self.starting_step, self.settings.steps_to_run):
            self.adjust_learning_rate(step)
            samples = next(train_dataset_generator)
            if len(samples) == 2:
                labeled_examples, labels = samples
                labeled_examples, labels = labeled_examples.to(gpu), labels.to(gpu)
            else:
                labeled_examples, primary_labels, secondary_labels = samples
                labeled_examples, labels = labeled_examples.to(gpu), (primary_labels.to(gpu), secondary_labels.to(gpu))
            self.dnn_training_step(labeled_examples, labels, step)
            if self.dnn_summary_writer.is_summary_step() or step == self.settings.steps_to_run - 1:
                print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                step_time_start = datetime.datetime.now()
                self.eval_mode()
                with torch.no_grad():
                    self.validation_summaries(step)
                self.train_mode()
            self.handle_user_input(step)
