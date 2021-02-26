
##### eeg library #####
from mne.filter import filter_data


def mi_band_pass_filter(eeg, frequency):
    # band-pass filter
    eeg = filter_data(eeg.T,
                      sfreq=frequency,
                      l_freq=7.0,
                      h_freq=35.0, picks=None, filter_length='auto',
                      l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
                      method='iir', iir_params=None, copy=True, phase='zero',
                      fir_window='hamming', fir_design='firwin',
                      pad='reflect_limited', verbose=None)
    return eeg.T
