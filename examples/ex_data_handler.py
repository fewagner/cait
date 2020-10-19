import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=35,
                        module='DetF',
                        channels=[26, 27],
                        record_length=16384)

    # the instance can convert the rdt file to a hdf5 file
    dh.convert_dataset(path_rdt='toy_data/run35_DetF/',
                       fname='bck_001',
                       path_h5='toy_data/run35_DetF/',
                       tpa_list=[0.],
                       calc_mp=True,
                       calc_fit=False,
                       calc_sev=False,
                       processes=4)

    # add labels
    dh.import_labels(path_labels='toy_data/')
