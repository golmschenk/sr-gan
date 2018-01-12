"""
General settings.
"""

class Settings():
    def __init__(self):
        self.trial_name = 'base'
        self.steps_to_run = 10000000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 100
        self.presentation_step_period = 1000
        self.summary_step_period = 1000
        self.labeled_dataset_size = 10
        self.unlabeled_dataset_size = 50000
        self.test_dataset_size = 1000
