from scipy import io
import numpy as np

from core.data import *
from collections import namedtuple


def load_mi_classification(type_data, load_train=True):
    """
    Load the Track#1 of OSF competition.
    link to dl : https://osf.io/s6t9e/
    #Sampling Rate 1000Hz
    #62 electrodes
    #20 subjects
    #4 secondes : [0 - 3] : black cross [3 - 4] : right/left hand task
    #10 trials*3
    :param type_data: Data class used for instantiation
    :param load_train: boolean, if true load the training data, otherwise load the test
    :return: array of data tuples
    """

    data = []
    if load_train:
        key_dict = 'Training'
        subpath = 'Training Set'
    else:
        key_dict = 'Test'
        subpath = 'Test Set'

    for subject in range(1, 21):
        path = '../data/OSF_Competition/Track#1 Few-shot EEG learning/{}/Data_Sample{:02d}.mat'\
            .format(subpath, subject)
        raw_data = io.loadmat(path)[key_dict][0][0]

        eeg = raw_data[0]  # [t,trial,n]
        nb_trials = eeg.shape[1]
        frequency = raw_data[1][0, 0]
        y_dec = raw_data[2][0]

        one_hot = raw_data[3]
        y_txt = raw_data[4]
        map_label = raw_data[5]
        chan = raw_data[6]
        y_dec = y_dec.astype(np.int)
        data.append(type_data(eeg=eeg,
                              nb_trials=nb_trials,
                              frequency=frequency,
                              y_dec=y_dec,
                              one_hot=one_hot,
                              y_txt=y_txt,
                              map_label=map_label,
                              chan=chan))
    return data



