"""
Code for accessing the data in the database easily.
"""

from scipy.io import loadmat
from datetime import datetime
import os


def calculate_age(taken, date_of_birth):
    birth_datetime = datetime.fromordinal(max(int(date_of_birth) - 366, 1))

    # Assume the photo was taken in the middle of the year
    if birth_datetime.month < 7:
        return taken - birth_datetime.year
    else:
        return taken - birth_datetime.year - 1


def get_database_meta(database_name):
    mat_path = "data/{}_crop/{}.mat".format(database_name, database_name)
    meta = loadmat(mat_path)
    full_paths = meta[database_name][0, 0]["full_path"][0]
    dobs = meta[database_name][0, 0]["dob"][0]
    genders = meta[database_name][0, 0]["gender"][0]
    photo_takens = meta[database_name][0, 0]["photo_taken"][0]
    face_scores = meta[database_name][0, 0]["face_score"][0]
    second_face_scores = meta[database_name][0, 0]["second_face_score"][0]
    ages = [calculate_age(photo_takens[i], dobs[i]) for i in range(len(dobs))]
    return full_paths, dobs, genders, photo_takens, face_scores, second_face_scores, ages
