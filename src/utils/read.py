from scipy import io
import numpy as np

from collections import namedtuple


def load_mi_classification():
    """
    Load the Track#1 of OSF competition.
    link to dl : https://osf.io/s6t9e/
    #Sampling Rate 1000Hz
    #62 electrodes
    #20 subjects
    #4 secondes : [0 - 3] : black cross [3 - 4] : right/left hand task
    #10 trials*3
    :return: array of data tuples
    """
    Data = namedtuple('Data',
                      ['eeg', 'nb_trials', 'frequency', 'y_dec', 'one_hot', 'y_txt', 'map_label', 'chan'])
    data = []
    key_dict = 'Training'
    for subject in range(1, 21):
        path = '../data/OSF_Competition/Track#1 Few-shot EEG learning/Training set/Data_Sample{:02d}.mat'\
            .format(subject)
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
        data.append(Data(eeg=eeg,
                         nb_trials=nb_trials,
                         frequency=frequency,
                         y_dec=y_dec,
                         one_hot=one_hot,
                         y_txt=y_txt,
                         map_label=map_label,
                         chan=chan))
    return data


