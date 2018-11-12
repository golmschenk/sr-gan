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

    def load_models(self):
        """Loads existing models if they exist at `self.settings.load_model_path`."""
        if self.settings.load_model_path:
            latest_dnn_model = None
            model_path_file_names = os.listdir(self.settings.load_model_path)
            for file_name in model_path_file_names:
                match = re.search(r'(DNN|D|G)_model_?(\d+)?\.pth', file_name)
                if match:
                    if match.group(1) == 'DNN':
                        latest_dnn_model = self.compare_model_path_for_latest(latest_dnn_model, match)
            latest_dnn_model = None if latest_dnn_model is None else latest_dnn_model.group(0)
            if not torch.cuda.is_available():
                map_location = 'cpu'
            else:
                map_location = None
            if latest_dnn_model:
                dnn_model_path = os.path.join(self.settings.load_model_path, latest_dnn_model)
                print('DNN model loaded from `{}`.'.format(dnn_model_path))
                self.DNN.load_state_dict(torch.load(dnn_model_path, map_location), strict=False)

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

    def save_models(self, step=None):
        """Saves the network models."""
        if step is not None:
            suffix = '_{}'.format(step)
        else:
            suffix = ''
        torch.save(self.DNN.state_dict(), os.path.join(self.trial_directory, 'DNN_model{}.pth'.format(suffix)))

    def training_loop(self):
        """Runs the main training loop."""
        train_dataset_generator = self.infinite_iter(self.train_dataset_loader)
        step_time_start = datetime.datetime.now()
        for step in range(self.settings.steps_to_run):
            self.adjust_learning_rate(step)
            labeled_examples, labels, knn_maps = next(train_dataset_generator)
            labeled_examples, labels, knn_maps = labeled_examples.to(gpu), labels.to(gpu), knn_maps.to(gpu)
            self.dnn_training_step(labeled_examples, labels, knn_maps, step)
            if self.dnn_summary_writer.is_summary_step() or step == self.settings.steps_to_run - 1:
                print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                step_time_start = datetime.datetime.now()
                self.eval_mode()
                self.validation_summaries(step)
                self.train_mode()
            self.handle_user_input(step)
