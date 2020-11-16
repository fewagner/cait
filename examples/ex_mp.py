import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cait as ai
from cait.features import calc_additional_parameters, distribution_skewness
import numpy as np

if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=35,
                        module='DetF',
                        channels=[26, 27],
                        record_length=16384)

    # set the file path for the hdf5 file
    dh.set_filepath(path_h5='./toy_data',
                    fname='bck_001')

    # f = dh.get_filehandle()
    #
    # event = f['events']['event'][0, 400]
    # of_real = np.array(f['optimumfilter']['optimumfilter_real'][0])
    # of_imag = np.array(f['optimumfilter']['optimumfilter_real'][0])
    # of = of_real + 1j*of_imag
    #
    # print(calc_additional_parameters(event,
    #                            optimal_transfer_function=of))

    # dh.recalc_additional_mp(type='events')

    dh.show_values(group='events',
                   key='add_mainpar',
                   idx0=0,
                   idx2=15,
                   bins=200,
                   range=[-0.001, 0.05])

    f = dh.get_filehandle()
    skews = f['events']['add_mainpar'][0, :, 15]
    print(np.max(skews))
    print(np.min(skews))