# imports

import numpy as np


# function

def write_xy_file(filepath, data, title, axis):
    """
    Writes a txt file in the XY format.

    The XY format is intended for plotting and data sharing. It has in the first line the title of the plot, followed
    by one line per axis with the axis label. Then the data follows, with one data point per line and one axis in each
    columns, seperated by tabulators.

    :param filepath: The path where we write the file.
    :type filepath: string
    :param data: The data that we want to plot
    :type data: array of shape (nmbr dimensions, nmbr data points)
    :param title: The title of the data plot.
    :type title: string
    :param axis: The axis labels.
    :type axis: list of strings
    """
    print('Start write.')

    dims = len(data)
    nmbr_values = len(data[0])

    if len(axis) != dims:
        raise KeyError('Number of dimensions of data array does not match number axis!')

    file = open(filepath, 'w')
    file.write(title + '\n')
    for s in axis:
        file.write(s + '\n')
    for i in range(nmbr_values):
        for j in range(dims):
            if j < dims - 1:
                file.write("{} \t".format(data[j][i]))
            else:
                file.write("{} \n".format(data[j][i]))
    file.close()

    print('File written.')


def read_xy_file(filepath, skip_lines=4, separator='\t'):
    """
    Reads a txt file in the XY format.

    The XY format is intended for plotting and data sharing. It has in the first line the title of the plot, followed
    by one line per axis with the axis label. Then the data follows, with one data point per line and one axis in each
    columns, seperated by tabulators.

    :param filepath: The path from where we read the file.
    :type filepath: string
    :param skip_lines: The number of lines at beginning of the file that contain no data (title, axis labels, ...).
    :type skip_lines: int
    :param separator: The unicode of the  column separator, default tabulator.
    :type separator: string
    :return: The data array.
    :rtype: array
    """
    values = np.genfromtxt(filepath,
                           skip_header=skip_lines,
                           delimiter=separator)

    print("File read.")

    return values
