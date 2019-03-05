"""
Code for displaying the results of a batch of experiments.
"""
import collections
import re

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import matplotlib2tikz
import pandas as pd
import glob
import os
import itertools
import math
import numpy as np
import seaborn as sns

latex_document_directory = 'logs'
sns.set()
dnn_color = sns.color_palette()[2]
gan_color = sns.color_palette()[3]
dggan_color = sns.color_palette()[4]


class Display:
    """
    A class for creating the display elements in latex
    """

    def __init__(self, calling_file, table_top_left_cell='', table_indexes_override=None):
        calling_script_directory = os.path.dirname(os.path.realpath(calling_file))
        self.latex_name = os.path.basename(calling_script_directory).replace(' ', '')
        self.logs = Log.create_all_in_directory(calling_script_directory)
        self.latex_document_directory = latex_document_directory
        self.table_columns_name = table_top_left_cell
        self.power_of_ten_format_on = True
        self.table_index_rename_dictionary = table_indexes_override
        self.table_axis_names = {'x': 'x-axis', 'y': 'y-axis'}
        self.generate_heatmap_pgfplots_colormap()

    def generate_heatmap_pgfplots_colormap(self):
        """Generates a file containing the latex code to create a colormap in pgfplots."""
        file_lines = ['\\pgfplotsset{colormap={heatmap colors}{\n']
        color_map = sns.cubehelix_palette(30, start=2.8, rot=0, dark=0.9, light=1, reverse=True)
        for color in color_map:
            file_lines.append('rgb=({},{},{})\n'.format(color[0], color[1], color[2]))
        file_lines.append('}}\n')
        with open(os.path.join(self.latex_document_directory, 'HeatmapColormapDefinition.tex'), 'w') as file:
            file.writelines(file_lines)

    def create_latex_error_plot_for_logs(self, get_parameter_function, secondary_sort_get_parameter_function):
        """
        Plot figure containing the comparison between two scalars for all TensorFlow event file logs in a directory.

        :param get_parameter_function: A function to get the parameter from the log to vary the plotting on.
        :type get_parameter_function: (Log) -> T
        :param secondary_sort_get_parameter_function: A second parameter getting function for grouping logs.
        :type secondary_sort_get_parameter_function: (Log) -> T
        """
        x_axis_scalar = 'Epoch'
        y_axis_scalar = '% Error'
        sns.set(style='whitegrid', font='serif')
        figure, axes = plt.subplots(num=None, figsize=(6, 4), dpi=150)
        plt.figure(figure.number)
        plt.sca(axes)
        axes.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:g}'))
        axes.grid(b=True, which='major')
        axes.grid(b=True, which='minor')
        color_count = len(set([get_parameter_function(log) for log in self.logs]))
        color_palette = sns.cubehelix_palette(color_count, start=0.4, rot=0.4, light=0.7)
        colors_cycle = itertools.cycle(color_palette)
        line_cycles = ['solid', 'dashed', 'dashdot', 'dotted']
        names_using_color = {}
        legend = []
        self.logs.sort(key=lambda log_: (get_parameter_function(log_),
                                         secondary_sort_get_parameter_function(log_)))
        for log in self.logs:
            line_styles_cycle = itertools.cycle(line_cycles)
            line_style = next(line_styles_cycle)
            match_value = str(get_parameter_function(log))
            if match_value in names_using_color:
                color = names_using_color[match_value]
                line_style = next(line_styles_cycle)
            else:
                color = next(colors_cycle)
                names_using_color[match_value] = color
            axes = log.scalars_data_frame.plot(x=x_axis_scalar, y=y_axis_scalar, logy=True, ax=axes, color=color,
                                               linestyle=line_style)
            axes.set(ylabel=y_axis_scalar.replace('% ', ''))
            legend.append(os.path.basename(os.path.dirname(log.event_file_name)))
        axes.legend(legend)
        if axes.legend():
            axes.legend().set_visible(False)
        xmin, xmax, ymin, ymax = axes.axis()
        ymin = math.floor(ymin)
        axes.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=100)
        axes.set_yticks([100, 10, ymin])
        plt.sca(axes)
        latex_file_name = os.path.join(self.latex_document_directory, 'Figures', self.latex_name + '.tex')
        extra_axis_parameters = {'tick style={draw=none}', 'yminorgrids',
                                 'y tick label style={/pgf/number format/1000 sep=\,}',
                                 'log base 10 number format code/.code={$\pgfmathparse{10^(#1)}' +
                                 '\pgfmathprintnumber{\pgfmathresult}$}',
                                 'extra y tick style={log identify minor tick positions=false}'}
        if ymin not in [1, 10]:
            extra_axis_parameters.add('extra y ticks={{{}}}'.format(ymin))
        matplotlib2tikz.save(latex_file_name, figure=figure,
                             extra_axis_parameters=extra_axis_parameters)
        delete_file_lines_starting_with(latex_file_name, 'legend entries=')
        file_string_replace(latex_file_name, 'dashed', 'dash pattern=on 5pt off 2pt')


class Log:
    """
    A class to be a simple wrapper for the log data.

    :type experiment: mnist_experiment.MnistExperiment
    :type scalars_data_frame: pandas.DataFrame
    """

    def __init__(self, event_file_name):
        self.experiment = None
        self.event_file_name = event_file_name
        self.directory_name = os.path.dirname(self.event_file_name)
        self.scalars_data_frame = None
        self.set_data_frame_from_event_file_scalar_summaries()

    def __str__(self):
        return self.directory_name

    @classmethod
    def create_all_in_directory(cls, directory, exclude_if_no_final_model_exists=False):
        """
        Generate all logs in directory.
        """
        glob_string = os.path.join(directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
        if exclude_if_no_final_model_exists:
            event_file_names = [event_file_name for event_file_name in event_file_names
                                if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(event_file_name)),
                                                               'D_model.pth'))]
        return [Log(event_file_name) for event_file_name in event_file_names]

    @classmethod
    def create_all_logs_for_current_directory(cls):
        """
        Generate all logs in current directory.

        :return: The list of logs.
        :rtype: list[Log]
        """
        return Log.create_all_in_directory('.')

    def set_data_frame_from_event_file_scalar_summaries(self):
        """
        Creates a Pandas data frame from a scalar summaries event file.

        :return: The scalars' data frame.
        :rtype: pd.DataFrame
        """

        summary_iterator = tf.train.summary_iterator(self.event_file_name)
        self.scalars_data_frame = pd.DataFrame()
        for event in summary_iterator:
            event_step = int(event.step)
            for value in event.summary.value:
                self.scalars_data_frame.at[event_step, value.tag] = value.simple_value
        self.scalars_data_frame.sort_index(inplace=True)


def power_of_ten_latex_format(value):
    """
    Convert the value into a power of 10 format of latex math.

    :param value: The value to give in power of 10 latex formatting.
    :type value: float
    :return: The latex formatted string.
    :rtype: str
    """
    if value == 0:
        return '{$0$}'
    power = math.log(value, 10)
    return '{{$10^{{{0:.3g}}}$}}'.format(power)


def delete_file_lines_starting_with(file_name, string_to_find):
    """
    Deletes the lines starting with the given string in a file.

    :param file_name: The name of the file to open.
    :type file_name: str
    :param string_to_find: The string to search for.
    :type string_to_find: str
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if not line.startswith(string_to_find)]
    with open(file_name, 'w') as file:
        file.writelines(lines)


def file_string_replace(file_name, old_string, new_string):
    """
    Replaces all instances of a string in a file with another string.

    :param file_name: The name of the file to open.
    :type file_name: str
    :param old_string: The original string to be replaced.
    :type old_string: str
    :param new_string: The string which will be added in.
    :type new_string: str
    """
    with open(file_name, 'r') as file:
        contents = file.read()
        contents = contents.replace(old_string, new_string)
    with open(file_name, 'w') as file:
        file.write(contents)


def plot_dnn_vs_gan_average_error_by_hyper_parameter(logs_directory, y_axis_label='Age MAE (years)',
                                                     x_axis_label='Labeled Dataset Size',
                                                     include_filter=None,
                                                     exclude_filter=None,
                                                     include_full_scatter=True,
                                                     match_hyper_parameter_regex=r' le(\d+) ',
                                                     hyper_parameter_type='int',
                                                     experiment_name='age', linthreshx=0.1,
                                                     x_axis_scale='symlog',
                                                     number_of_elements_to_average=3):
    """Plots DNN vs GAN errors based on a hyperparameter."""
    alpha = 0.2
    logs = Log.create_all_in_directory(logs_directory, exclude_if_no_final_model_exists=True)
    dnn_results = collections.defaultdict(list)
    gan_results = collections.defaultdict(list)
    dnn_points = []
    gan_points = []
    for log in logs:
        if include_filter is not None and not re.search(include_filter, log.event_file_name):
            continue
        if exclude_filter is not None and re.search(exclude_filter, log.event_file_name):
            continue
        match_hyper_parameter = re.search(match_hyper_parameter_regex, log.event_file_name).group(1)
        if hyper_parameter_type == 'int':
            match_hyper_parameter = int(match_hyper_parameter)
        elif hyper_parameter_type == 'float':
            match_hyper_parameter = float(match_hyper_parameter)
        model_type = re.search(r'/(DNN|GAN)/', log.event_file_name).group(1)
        last_errors = log.scalars_data_frame.iloc[-number_of_elements_to_average:]['1_Validation_Error/MAE'].tolist()
        error = np.nanmean(last_errors)
        if model_type == 'GAN':
            gan_results[match_hyper_parameter].append(error)
            gan_points.append((match_hyper_parameter, error))
        else:
            dnn_results[match_hyper_parameter].append(error)
            dnn_points.append((match_hyper_parameter, error))
    average_dnn_results = {example_count: np.mean(values) for example_count, values in dnn_results.items()}
    average_gan_results = {example_count: np.mean(values) for example_count, values in gan_results.items()}
    dnn_plot_x, dnn_plot_y = zip(*sorted(average_dnn_results.items()))
    gan_plot_x, gan_plot_y = zip(*sorted(average_gan_results.items()))
    dnn_plot_y, gan_plot_y = np.array(dnn_plot_y), np.array(gan_plot_y)
    figure, axes = plt.subplots()
    if x_axis_scale == 'symlog':
        axes.set_xscale(x_axis_scale, linthreshx=linthreshx)
    else:
        axes.set_xscale(x_axis_scale)
    axes.set_xticks(dnn_plot_x)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel(x_axis_label)
    axes.set_ylabel(y_axis_label)
    if include_full_scatter:
        axes.scatter(*np.array(dnn_points).transpose(), color=dnn_color, alpha=alpha)
        axes.scatter(*np.array(gan_points).transpose(), color=gan_color, alpha=alpha)
    axes.plot(dnn_plot_x, dnn_plot_y, color=dnn_color, label='DNN')
    axes.plot(gan_plot_x, gan_plot_y, color=gan_color, label='GAN')
    axes.legend().set_visible(True)
    matplotlib2tikz.save(os.path.join('latex', '{}-gan-vs-dnn.tex'.format(experiment_name)))
    plt.show()
    plt.close(figure)
    figure, axes = plt.subplots()
    if x_axis_scale == 'symlog':
        axes.set_xscale(x_axis_scale, linthreshx=linthreshx)
    else:
        axes.set_xscale(x_axis_scale)
    axes.set_xticks(dnn_plot_x)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel(x_axis_label)
    axes.set_ylabel('GAN to DNN Relative Error')
    if include_full_scatter:
        relative_points = [(dnn_point[0], gan_point[1]/dnn_point[1])
                           for dnn_point, gan_point in zip(dnn_points, gan_points)]
        axes.scatter(*np.array(relative_points).transpose(), color=gan_color, alpha=alpha)
    axes.plot(gan_plot_x, gan_plot_y / dnn_plot_y, color=gan_color)
    matplotlib2tikz.save(os.path.join('latex', '{}-gan-to-dnn-relative-error.tex'.format(experiment_name)))
    plt.show()
    plt.close(figure)


def plot_coefficient_dnn_vs_gan_error_over_training(single_log_directory):
    """Plots error over training comparing DNN to GAN."""
    logs = Log.create_all_in_directory(single_log_directory)
    if re.search(r'/GAN/', logs[0].event_file_name):
        gan_log, dnn_log = logs[0], logs[1]
    else:
        dnn_log, gan_log = logs[0], logs[1]
    figure, axes = plt.subplots()
    axes.set_xlabel('Training Step')
    axes.set_ylabel('MAE')
    dnn_log.scalars_data_frame.plot(y='1_Validation_Error/MAE', ax=axes, label='DNN', color=dnn_color)
    gan_log.scalars_data_frame.plot(y='1_Validation_Error/MAE', ax=axes, label='GAN', color=gan_color)
    matplotlib2tikz.save(os.path.join('latex', 'error-over-training.tex'))
    plt.show()
    plt.close(figure)


def average_values_of_trials(logs_directory, value_name, include_filter=None, exclude_filter=None):
    """Returns the average values during training steps for a set of trials matching the conditions."""
    logs = Log.create_all_in_directory(logs_directory, exclude_if_no_final_model_exists=True)
    trials_values_list = []
    for log in logs:
        if include_filter is not None and not re.search(include_filter, log.event_file_name):
            continue
        if exclude_filter is not None and re.search(exclude_filter, log.event_file_name):
            continue
        single_trial_values = log.scalars_data_frame.iloc[:][value_name]
        trials_values_list.append(single_trial_values)
    average_values = np.mean(trials_values_list, axis=0)
    return average_values


def get_summary_steps_in_first_log(logs_directory, include_filter=None, exclude_filter=None):
    """Returns the average values during training steps for a set of trials matching the conditions."""
    logs = Log.create_all_in_directory(logs_directory, exclude_if_no_final_model_exists=True)
    steps = np.array([])
    for log in logs:
        if include_filter is not None and not re.search(include_filter, log.event_file_name):
            continue
        if exclude_filter is not None and re.search(exclude_filter, log.event_file_name):
            continue
        steps = log.scalars_data_frame.index.values
        break
    return steps


if __name__ == '__main__':
    logs_directory_ = '/Users/golmschenk/Desktop/logs'
    value_name_ = '1_Validation_Error/MAE'
    steps_ = get_summary_steps_in_first_log(logs_directory_)
    log_values = average_values_of_trials(logs_directory_, value_name_, include_filter='abs_plus_one_log_mean_neg')
    sqrt_values = average_values_of_trials(logs_directory_, value_name_, include_filter='abs_plus_one_sqrt_mean_neg')
    linear_values = average_values_of_trials(logs_directory_, value_name_, include_filter='abs_mean_neg')
    figure, axes = plt.subplots()
    axes.set_yscale('log')
    axes.plot(steps_, log_values, color=sns.color_palette()[5], label='Replace log')
    axes.plot(steps_, sqrt_values, color=sns.color_palette()[6], label='Replace sqrt')
    axes.plot(steps_, linear_values, color=sns.color_palette()[7], label='Replace linear')
    axes.legend().set_visible(True)
    matplotlib2tikz.save(os.path.join('latex', 'loss-function-comparison.tex'))
    plt.show()
    plt.close(figure)
