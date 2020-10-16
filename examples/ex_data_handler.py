
import sys
import os

print('[change directory]')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

from cait import DataHandler

if __name__ == '__main__':

    # create Instance of the DataHandler Class
    dh = DataHandler(run=35,
                     module='DetF',
                     record_length=16384,
                     nmbr_channels=2)

    # the instance can convert the rdt file to a hdf5 file
    dh.convert_dataset(path_rdt='toy_data/Run35_DetF/',
                       fname='bck_001',
                       path_h5='toy_data/Run35_DetF/',
                       channels=[26, 27],
                       tpa_list=[0.],
                       calc_mp=True,
                       calc_fit=False,
                       calc_sev=False,
                       processes=4)

