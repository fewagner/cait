# imports

import numpy as np


# function

def write_xy_file(filepath, xdata, ydata, title, xaxis, yaxis):
    # TODO
    print('Start write.')

    file = open(filepath, 'w')
    file.write(title + '\n')
    file.write(xaxis + '\n')
    file.write(yaxis + '\n')
    for x, y in zip(xdata, ydata):
        file.write("{} \t{} \n".format(x, y))
    file.close()

    print('File written.')


def read_xy_file(filepath, skip_lines=4, separator='\t'):
    # TODO
    values = np.genfromtxt(filepath,
                           skip_header=skip_lines,
                           delimiter=separator)

    print("File read.")

    return values
