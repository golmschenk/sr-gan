"""
Script to clean log folder quickly.
"""
import shutil
import os

def clear_logs():
    shutil.rmtree('logs')
    os.makedirs('logs')


if __name__ == '__main__':
    clear_logs()