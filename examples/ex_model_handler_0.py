"""
DEPRICATED!!
This is an example of how to train a machine learning model on all channels

See also the How-Tos on the Wiki page!
https://git.cryocluster.org/fwagner/cait/-/wikis/4.3-Train-and-Evaluate-ML-Models
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cait as ai

if __name__ == '__main__':

    # create instance of ModelHandler
    mh = ai.ModelHandler(run=35,
                         module='DetF',
                         model_type='rf',
                         record_length=16384,
                         nmbr_channels=2,
                         down=64)

    # add information about the data we use
    mh.add_data(data_path='toy_data/',
                bcks=['001'],
                module_channels=[26, 27])

    # create classifiers for phonon and light channel
    for c in [0, 1]:
        mh.add_classifier(channel=c,
                          down=64)

    # train the classifiers
    for c in [0, 1]:
        mh.train_classifier(channel=c,
                            test_size=0.3,
                            random_seed=42)

    # save the ModelHandler
    mh.save(path='toy_models')