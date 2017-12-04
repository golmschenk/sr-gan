"""
Script to clean log folder quickly.
"""
import shutil
import os

shutil.rmtree('logs')
os.makedirs('logs')
