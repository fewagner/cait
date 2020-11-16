"""
This is an example of how to open a HDF5 data set and label it.
See also the How-Tos on the Wiki page!
https://git.cryocluster.org/fwagner/cait/-/wikis/4.2-How-To:-View-and-Label-Events
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':

    # create instance of EventInterface
    ei = ai.EventInterface(module='TUM38',
                        run=33,
                        record_length=8192,
                        sample_frequency=25000,
                        nmbr_channels=2,
                        down=1)

    # load a bck file
    ei.load_bck(path='toy_data/',
                bck_nmbr='013',
                channels=['36', '37'],
                bck_naming='bck',
                which_to_label=['events'])

    # create a csv file with labels
    # alternatively you could load a labels file
    # Attention! this overwrites an existing csv file!
    ei.create_labels_csv(path='toy_data/')

    ei.load_of()

    ei.load_sev_par()

    # start the labeling process interface
    # you could also look at an individual event by index
    ei.start_labeling(start_from_idx=1,
                      label_only_class=None)