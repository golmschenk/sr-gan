"""
General settings.
"""

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

        self.histogram_logging = False