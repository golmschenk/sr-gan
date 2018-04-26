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
import seaborn as sns
import glob
import os
import itertools
import math
import pickle
import numpy as np
import importlib.util
import seaborn as sns

latex_document_directory = 'logs'
sns.set()
dnn_color = sns.color_palette()[3]
gan_color = sns.color_palette()[2]

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

    def create_table_for_logs(self, get_column_parameter_function, get_row_parameter_function,
                              row_is_numeric_parameter=False):
        """
        Generate a table for the logs.
        """
        x_value_set = set()
        y_value_set = set()
        for log in self.logs:
            x_value_set.add(get_column_parameter_function(log))
            y_value_set.add(get_row_parameter_function(log))
        x_values = sorted(list(x_value_set))
        y_values = sorted(list(y_value_set))
        data_frame = pd.DataFrame(index=y_values, columns=x_values, dtype=np.float32)
        for log in self.logs:
            index = get_row_parameter_function(log)
            column = get_column_parameter_function(log)
            data_frame[column][index] = log.experiment.test_error * 100
        self.generate_latex_table_from_data_frame(data_frame, row_is_numeric_parameter=row_is_numeric_parameter)

    def create_table_including_lowest_validation_for_logs(self, get_column_parameter_function,
                                                          get_row_parameter_function=None):
        """
        Generate a table for the logs.
        """
        x_value_set = set()
        y_value_set = set()
        for log in self.logs:
            x_value_set.add(get_column_parameter_function(log))
            if get_row_parameter_function:
                y_value_set.add(get_row_parameter_function(log))
        x_values = sorted(list(x_value_set))
        if y_value_set:
            y_values = [str(row_parameter) + ' ' + error_name for row_parameter in sorted(list(y_value_set))
                        for error_name in ['test error', 'lowest validation error']]
        else:
            y_values = ['Test error', 'Lowest validation error']
        data_frame = pd.DataFrame(index=y_values, columns=x_values, dtype=np.float32)
        for log in self.logs:
            column = get_column_parameter_function(log)
            if get_row_parameter_function:
                index = str(get_row_parameter_function(log)) + ' test error'
            else:
                index = 'Test error'
            data_frame[column][index] = log.experiment.test_error * 100
            if get_row_parameter_function:
                index = str(get_row_parameter_function(log)) + ' lowest validation error'
            else:
                index = 'Lowest validation error'
            data_frame[column][index] = log.scalars_data_frame['Error'].min() * 100
        self.generate_latex_table_from_data_frame(data_frame)

    def generate_latex_table_from_data_frame(self, data_frame, row_is_numeric_parameter=False):
        """
        Generates a latex table file from a data frame.

        :param data_frame: The data frame to generate the table from.
        :type data_frame: pd.DataFrame
        """
        if self.power_of_ten_format_on:
            data_frame.columns = map(power_of_ten_latex_format, data_frame.columns)
        data_frame = data_frame.applymap(lambda x: '%.2f' % x)
        # noinspection PyTypeChecker
        column_format = 'l' + ('S' * len(data_frame.columns))
        if self.table_index_rename_dictionary:
            # noinspection PyUnresolvedReferences
            data_frame.rename(index=self.table_index_rename_dictionary, inplace=True)
        elif data_frame.shape[0] == 1:
            data_frame.index = ['Test Error']
        if row_is_numeric_parameter:
            data_frame.index = map(power_of_ten_latex_format, data_frame.index)
        else:
            data_frame.index = map(lambda string: '{' + string + '}', data_frame.index)
        data_frame.columns.names = ['{' + self.table_columns_name + '}']
        latex_tabular = self.data_frame_to_pgfplotstable_tex(data_frame)
        with open(os.path.join(self.latex_document_directory, 'Tables', self.latex_name + '.tex'), 'w') as latex_file:
            latex_file.write(latex_tabular)

    def data_frame_to_pgfplotstable_tex(self, data_frame):
        """
        Creates the tex code for the pgfplotstable for the data frame.

        :param data_frame: The data frame to generate the pgfplotstable for.
        :type data_frame: pandas.DataFrame
        :return: The string of the pgfplotstable code.
        :rtype: str
        """
        csv_latex_relative_name = 'Tables/' + self.latex_name + '.csv'
        csv_file_name = os.path.join(self.latex_document_directory, 'Tables', self.latex_name + '.csv')
        data_frame.to_csv(path_or_buf=csv_file_name, index_label=data_frame.columns.names[0])
        contents = ('\pgfplotstabletypeset[\n' +
                    'color cells={{min=1,max=100}},\n' +
                    #'color cells={{min={},max={}}},\n'.format(data_frame.values.astype(np.float).min(),
                    #                                          data_frame.values.astype(np.float).max()) +
                    'col sep=comma\n' +
                    ']{' +
                    csv_latex_relative_name +
                    '}\n')
        return contents


class Log:
    """
    A class to be a simple wrapper for the log data.

    :type experiment: mnist_experiment.MnistExperiment
    :type scalars_data_frame: pandas.DataFrame
    """

    def __init__(self, event_file_name):
        self.event_file_name = event_file_name
        self.directory_name = os.path.dirname(self.event_file_name)
        self.scalars_data_frame = None
        self.set_data_frame_from_event_file_scalar_summaries()

    def __str__(self):
        return self.directory_name

    @classmethod
    def create_all_in_directory(cls, directory):
        """
        Generate all logs in directory.

        :param directory: The directory to recursively search for log files in.
        :type directory: str
        :return: The list of logs.
        :rtype: list[Log]
        """
        glob_string = os.path.join(directory, '**', 'events.out.tfevents*')
        event_file_names = glob.glob(glob_string, recursive=True)
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


def plot_dnn_vs_gan_average_error_by_labeled_dataset_size(logs_directory):
    logs = Log.create_all_in_directory(logs_directory)
    dnn_results = collections.defaultdict(list)
    gan_results = collections.defaultdict(list)
    for log in logs:
        labeled_dataset_size = int(re.search(r' le(\d+) ', log.event_file_name).group(1))
        model_type = re.search(r'/(DNN|GAN)/', log.event_file_name).group(1)
        last_errors = log.scalars_data_frame.iloc[-3:]['1_Validation_Error/MAE'].tolist()
        error = np.nanmean(last_errors)
        if model_type == 'GAN':
            gan_results[labeled_dataset_size].append(error)
        else:
            dnn_results[labeled_dataset_size].append(error)
    average_dnn_results = {example_count: np.mean(values) for example_count, values in dnn_results.items()}
    average_gan_results = {example_count: np.mean(values) for example_count, values in gan_results.items()}
    dnn_plot_x, dnn_plot_y = zip(*sorted(average_dnn_results.items()))
    gan_plot_x, gan_plot_y = zip(*sorted(average_gan_results.items()))
    dnn_plot_y, gan_plot_y = np.array(dnn_plot_y), np.array(gan_plot_y)
    figure, axes = plt.subplots()
    axes.set_xscale('log')
    axes.set_xticks(dnn_plot_x)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel('Labeled Dataset Size')
    axes.set_ylabel('Age MAE')
    axes.plot(dnn_plot_x, dnn_plot_y, color=dnn_color, label='DNN')
    axes.plot(gan_plot_x, gan_plot_y, color=gan_color, label='GAN')
    axes.legend().set_visible(True)
    matplotlib2tikz.save(os.path.join('latex', 'dnn-vs-gan.tex'))
    plt.show()
    plt.close(figure)
    figure, axes = plt.subplots()
    axes.set_xscale('log')
    axes.set_xticks(dnn_plot_x)
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel('Labeled Dataset Size')
    axes.set_ylabel('GAN to DNN Relative Error')
    axes.plot(gan_plot_x, gan_plot_y / dnn_plot_y, color=gan_color)
    matplotlib2tikz.save(os.path.join('latex', 'gan-to-dnn-relative-error.tex'))
    plt.show()
    plt.close(figure)


def plot_coefficient_dnn_vs_gan_error_over_training(single_log_directory):
    logs = Log.create_all_in_directory('logs')
    if re.search(r'/GAN/', logs[0].event_file_name):
        gan_log, dnn_log = logs[0], logs[1]
    else:
        dnn_log, gan_log = logs[0], logs[1]
    figure, axes = plt.subplots()
    gan_log.scalars_data_frame.plot(y='1_Validation_Error/MAE', ax=axes, label='GAN', color=gan_color)
    dnn_log.scalars_data_frame.plot(y='1_Validation_Error/MAE', ax=axes, label='DNN', color=dnn_color)
    plt.show()



if __name__ == '__main__':
    plot_dnn_vs_gan_average_error_by_labeled_dataset_size('/Users/golmschenk/Desktop/Preliminary Age Results')
