import numpy as np

class Data:
    def __init__(self, eeg, nb_trials, frequency, y_dec, one_hot, y_txt, map_label, chan):
        self._eeg = eeg #[step, trial, electrode]
        self._nb_trials = nb_trials
        self._frequency = frequency
        self._y_dec = y_dec
        self._one_hot = one_hot
        self._y_txt = y_txt
        self._map_label = map_label
        self._chan = chan
        self._features = []

    def freq_filter(self, filter_fn):
        self._eeg = filter_fn(self._eeg)

    def spatial_filter(self, filter_fn):
        self._eeg, self._chan = filter_fn(self._eeg, self._chan)

    def compute_features(self):
        for trial in self._nb_trials:
            self._features.append(np.cov(self._eeg[:, trial, :].T.reshape(len(self._chan), -1)).flatten())

    @property
    def features(self):
        return self._features
