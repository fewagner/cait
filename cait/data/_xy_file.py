# imports

import numpy as np


# function

def write_xy_file(filepath, data, title, axis):
    """


    :param filepath:
    :type filepath:
    :param data:
    :type data:
    :param title:
    :type title:
    :param axis:
    :type axis:
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
    # TODO
    values = np.genfromtxt(filepath,
                           skip_header=skip_lines,
                           delimiter=separator)

    print("File read.")

    return values
