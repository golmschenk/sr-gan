"""
General settings.
"""

class Settings():
    def __init__(self):
        self.trial_name = 'base rg every layer norm noise 50e l2u normalized mean features fl 1e-1'
        self.steps_to_run = 10000000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 100
        self.presentation_step_period = 1000
        self.summary_step_period = 1000
        self.labeled_dataset_size = 50
        self.unlabeled_dataset_size = 50000
        self.test_dataset_size = 1000
