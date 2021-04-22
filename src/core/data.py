##### generic library #####
import numpy as np
import abc
import matplotlib.pyplot as plt

##### features computation #####
import pywt

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
        self._eeg = filter_fn(self._eeg, self._frequency)

    def spatial_filter(self, filter_fn):
        self._eeg, self._chan = filter_fn(self._eeg, self._chan)

    def window_crop(self, stamp_start, stamp_stop):
        self._eeg = self._eeg[stamp_start:stamp_stop, :, :]

    @abc.abstractmethod
    def compute_features(self):
        pass

    @property
    def features(self):
        return np.array(self._features)

    @property
    def y_dec(self):
        return self._y_dec

    def plot_eeg(self, label, trial_to_show=3):
        eeg_to_plot = self._eeg[:, self._y_dec == label, :]
        eeg_to_plot = eeg_to_plot[:, :trial_to_show, :]
        t = np.arange(0, eeg_to_plot.shape[0])
        for i in range(eeg_to_plot.shape[1]):
            for j in range(eeg_to_plot.shape[2]):
                ax = plt.subplot(eeg_to_plot.shape[1], eeg_to_plot.shape[2], i * eeg_to_plot.shape[2] + j + 1)
                plt.plot(t, eeg_to_plot[:, i, j])
                plt.xlim(0, eeg_to_plot.shape[0])
        plt.show()

class CovData(Data):
    def compute_features(self):
        for trial in range(self._nb_trials):
            self._features.append(np.cov(self._eeg[:, trial, :].T.reshape(len(self._chan), -1)).flatten())


class TimeFrequencyData(Data):
    """
    Event-Related Desynchronization (ERD) May Not be Correlated with Motor Imagery BCI Performance
    The effects of handedness on sensorimotor rhythm desynchronization and motor-imagery BCI control
    """
    def compute_features(self):
        for trial in range(self._nb_trials):
            scales = np.arange(1, 128)
            print(self.y_dec[trial])
            sampling_period = 1.0/1000.0
            cwtmatr, freqs = pywt.cwt(self._eeg[:, trial, 5], scales, 'cmor3-3', sampling_period=sampling_period)
            power = (abs(cwtmatr)) ** 2

            plt.subplot(2, 1, 1)
            plt.imshow(power[:, :], aspect='auto')

            if self.y_dec[trial] == 2: #right
                plot_title = 'ERD/ERC C3'
            else: #left
                plot_title = 'ERD/ERC C4'

            plt.title('C3 ' + plot_title)
            plt.subplot(2, 1, 2)
            cwtmatr, freqs = pywt.cwt(self._eeg[:, trial, 7], scales, 'cmor3-3', sampling_period=sampling_period)
            power = (abs(cwtmatr)) ** 2
            plt.imshow(power[:, :], aspect='auto')
            plt.title('C4 ' + plot_title)
            plt.show()

