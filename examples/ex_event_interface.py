
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':

    # create instance of EventInterface
    ei = ai.EventInterface(module='DetF',
                        run=35,
                        record_length=16384,
                        sample_frequency=25000,
                        nmbr_channels=2,
                        down=1)

    # load a bck file
    ei.load_bck(path='toy_data/',
                bck_nmbr='001',
                channels=['26', '27'],
                bck_naming='bck',
                which_to_label=['events'])

    # create a csv file with labels
    # alternatively you could load a labels file
    ei.create_labels_csv(path='toy_data/')

    # start the labeling process interface
    # you could also look at an individual event by index
    ei.start_labeling(start_from_idx=1,
                      label_only_class=None)