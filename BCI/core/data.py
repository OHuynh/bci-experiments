##### generic library #####
import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy import signal

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

    def decomposition(self, decomposition_fn):
        self._eeg = decomposition_fn(self)

    @abc.abstractmethod
    def compute_features(self):
        pass

    @property
    def features(self):
        return np.array(self._features)

    @property
    def y_dec(self):
        return self._y_dec

    @property
    def eeg(self):
        return self._eeg

    @property
    def sampling_frequency(self):
        return self._frequency

    def plot_eeg(self, label, mode='raw', trial_to_show=5, channels_to_show=10):
        eeg_to_plot = self._eeg[:, self._y_dec == label, :]
        eeg_to_plot = eeg_to_plot[:, :trial_to_show, :min(channels_to_show, eeg_to_plot.shape[2])]
        t = np.arange(0, eeg_to_plot.shape[0])
        if mode == 'raw':
            for i in range(eeg_to_plot.shape[2]):
                for j in range(eeg_to_plot.shape[1]):
                    ax = plt.subplot(eeg_to_plot.shape[1], eeg_to_plot.shape[2], j * eeg_to_plot.shape[2] + i + 1)
                    plt.plot(t, eeg_to_plot[:, j, i])
                    plt.xlim(0, eeg_to_plot.shape[0])
                    plt.ylabel('trials')
                    plt.xlabel('channels')
            plt.show()
        elif mode == 'psd':
            samples = eeg_to_plot.shape[0]
            time_window = 1
            overlap_window = 0.75
            for i in range(eeg_to_plot.shape[1]):
                for j in range(eeg_to_plot.shape[2]):
                    f, Pxx_den = signal.welch(eeg_to_plot[:, i, j],
                                              self._frequency,
                                              nperseg=int(time_window * self._frequency**2 / samples),
                                              noverlap=int(overlap_window * self._frequency**2 / samples))
                    plt.semilogy(f, Pxx_den)
                    plt.xlabel('frequency [Hz]')
                    plt.ylabel('PSD [V**2/Hz]')
                    plt.show()

class TimeSeriesData(Data):
    def compute_features(self):
        self._features = self._eeg


class CovData(Data):
    def compute_features(self):
        for trial in range(self._nb_trials):
            self._features.append(np.cov(self._eeg[:, trial, :].T.reshape(self._eeg.shape[2], -1)).flatten())


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
            cwtmatr, freqs = pywt.cwt(self._eeg[:, trial, 0], scales, 'cmor3-3', sampling_period=sampling_period)
            power = (abs(cwtmatr)) ** 2

            plt.subplot(2, 1, 1)
            plt.imshow(power[:, :], aspect='auto')

            if self.y_dec[trial] == 2: #right
                plot_title = 'ERD/ERC C3'
            else: #left
                plot_title = 'ERD/ERC C4'

            plt.title('C3 ' + plot_title)
            plt.subplot(2, 1, 2)
            cwtmatr, freqs = pywt.cwt(self._eeg[:, trial, 1], scales, 'cmor3-3', sampling_period=sampling_period)
            power = (abs(cwtmatr)) ** 2
            plt.imshow(power[:, :], aspect='auto')
            plt.title('C4 ' + plot_title)
            plt.show()

