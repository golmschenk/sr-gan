"""
General settings.
"""

class Settings():
    def __init__(self):
        self.trial_name = 'fake generator L2 weight decay lr 1e-5'
        self.steps_to_run = 200000000
        self.temporary_directory = 'temporary'
        self.logs_directory = 'logs'
        self.batch_size = 100
        self.presentation_step_period = 1000
        self.summary_step_period = 1000
