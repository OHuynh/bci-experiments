from scipy import io
import numpy as np

from core.data import *
from collections import namedtuple

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

def load_osf_mi_classification(type_data, load_train=True):
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
        chan = raw_data[6].squeeze()
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


def load_eegbci_mi_classification(type_data, path):
    """
    EEGBCI dataset [1]
    [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
        Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer
        Interface (BCI)
    :return: array of data tuples
    """

    data = []
    for subject in range(1, 21):
        tmin, tmax = -1., 4.
        event_id = dict(hand_left=2, hand_right=3)
        runs = [4, 8, 12]   # Motor imagery: left vs right hand

        raw_files = [
            read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs, path=path)
        ]
        raw = concatenate_raws(raw_files)

        picks = pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False)
        labels = epochs.events[:, -1] - 2
        data.append(type_data(eeg=np.transpose(epochs.get_data(), [2, 0, 1]),
                              nb_trials=45,
                              frequency=raw.info['sfreq'],
                              y_dec=labels,
                              one_hot=None,
                              y_txt=None,
                              map_label=None,
                              chan=np.array([str.upper(chan.replace('.', '')) for chan in epochs.ch_names])))
    return data
