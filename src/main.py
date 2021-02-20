
##### generic library #####
from sklearn.svm import SVC
import scipy.linalg

##### eeg library #####
from mne.filter import filter_data

##### project import #####
from filters.spatial import *
from utils.read import *

def main():
    data = load_mi_classification()
    for subject in range(len(data)):
        data[subject] = mi_active_electrodes(data[subject])

        #band-pass filter
        for i in range(data[subject].eeg.shape[1]):
            data[subject].eeg = filter_data(data[subject].eeg.T,
                                            sfreq=1000.0,
                                            l_freq=7.0,
                                            h_freq=35.0, picks=None, filter_length='auto',
                                            l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
                                            method='iir', iir_params=None, copy=True, phase='zero',
                                            fir_window='hamming', fir_design='firwin',
                                            pad='reflect_limited', verbose=None)


if __name__ == "__main__":
    main()
