"""
This is an example how to convert an rdt file to a HDF5 dataset.

See also the How-Tos on the Wiki page!
https://git.cryocluster.org/fwagner/cait/-/wikis/4.1-How-To:-Convert-an-*.rdt-file-to-an-hdf5-dataset
"""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=33,
                        module='TUM38',
                        channels=[36, 37],
                        record_length=8192)

    # the instance can convert the rdt file to a hdf5 file
    dh.convert_dataset(path_rdt='toy_data/run33_TUM38/',
                       fname='bck_013',
                       path_h5='toy_data/run33_TUM38/',
                       tpa_list=[0., -1],
                       calc_mp=True,
                       calc_fit=False,
                       calc_sev=False,
                       processes=4)
