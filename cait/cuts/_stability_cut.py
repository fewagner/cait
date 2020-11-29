# imports

import numpy as np

# functions

def stability_cut(f, type, channel, idx):

    tpas = f['testpulses']['testpulseamplitude']
    tphs = f['testpulses']['mainpar'][channel, :, 0]  # 0 is the mainpar index for pulseheight

    unique_tpas = np.unique(tpas)

    # first very rough cleaning
    medians = []
    for val in unique_tpas:
        medians.append(np.median(tpas[tpas == val]))

        # get all hours that are +- median/4 away

    # now a bit finer cleaning
    means = []
        # get all hours that are not within 2 standard deviations

    # somehow kick all hours that are between the hours derived above