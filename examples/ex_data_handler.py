import cait as ai
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    # create Instance of the DataHandler Class
    dh = ai.DataHandler(run=35,
                        module='DetF',
                        channels=[26, 27],
                        record_length=16384)

    # the instance can convert the rdt file to a hdf5 file
    dh.convert_dataset(path_rdt='toy_data/',
                       fname='bck_001',
                       path_h5='toy_data/',
                       tpa_list=[0.],
                       calc_mp=True,
                       calc_fit=False,
                       calc_sev=False,
                       processes=4)

    # add labels
    dh.import_labels(path_labels='toy_data/')
