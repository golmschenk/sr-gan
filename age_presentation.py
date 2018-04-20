"""
Code for preparing and presenting data and results.
"""
import random

import matplotlib.pyplot as plt
import matplotlib2tikz as matplotlib2tikz
import seaborn as sns
import os

import shutil

from age_data import get_database_meta


def generate_age_distribution_image():
    sns.set()
    sns.set_style('dark')
    figure, axes = plt.subplots()
    full_paths, dobs, genders, photo_takens, face_scores, second_face_scores, ages = get_database_meta('imdb')
    image_paths = []
    for i in range(len(face_scores)):
        if 1.0 < face_scores[i]:
            image_paths.append(full_paths[i][0])
    print("IMDb images with scores over 1.0: {}".format(len(image_paths)))
    axes = sns.kdeplot(ages, clip=(0, 100), ax=axes, label='IMDB')
    imdb_ages = ages

    full_paths, dobs, genders, photo_takens, face_scores, second_face_scores, ages = get_database_meta('wiki')
    image_paths = []
    for i in range(len(face_scores)):
        if 1.0 < face_scores[i]:
            image_paths.append(full_paths[i][0])
    print("Wikipedia images with scores over 1.0: {}".format(len(image_paths)))
    axes = sns.kdeplot(ages, clip=(0, 100), ax=axes, label='WIKI')
    wiki_ages = ages

    axes = sns.kdeplot(imdb_ages + wiki_ages, clip=(0, 100), ax=axes, label='IMDB-WIKI')

    #axes.set_yticks([])
    axes.yaxis.set_major_locator(plt.NullLocator())
    #plt.show()
    matplotlib2tikz.save(os.path.join('latex', 'age-distribution.tex'))
    plt.close(figure)


def select_random_face_images(count=30):
    full_paths, dobs, genders, photo_takens, face_scores, second_face_scores, ages = get_database_meta('imdb')
    image_paths = []
    for i in range(len(face_scores)):
        if 1.0 < face_scores[i]:
            image_paths.append(full_paths[i][0])
    random.shuffle(image_paths)
    selected_image_paths = image_paths[:count]
    for index, selected_image_path in enumerate(selected_image_paths):
        shutil.copyfile('data/imdb_crop/{}'.format(selected_image_path), 'latex/example_images/{}.jpg'.format(index))


if __name__ == '__main__':
    select_random_face_images()
