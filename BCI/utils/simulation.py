import numpy as np


def mixt_mod(n=2, t=1024, K=5):
    s = np.zeros((n, t))
    for r in range(0, n):
        ind = randperm(t)
        s[r, ind[ind[0:K]]] = np.random.randn(K)

    return s


def make_sources(n_s=1, t_samp=1024, k=8, w=2):
    s = mixt_mod(n_s, t_samp, k)

    x = np.linspace(1, t_samp, t_samp) - t_samp / 2

    kern = np.exp(-abs(x) / (w / np.log(2)))
    kern = kern / np.max(kern)

    for r in range(0, n_s):
        s[r, :] = np.convolve(s[r, :], kern, mode='same')

    return s


def randperm(n=1):
    x = np.random.randn(n)
    i = x.argsort()
    return i


def generate_wavelet_sources(type_data,
                             start=1.0,
                             end=5.0,
                             pattern_len_max=2,
                             pattern_len_min=0.5,
                             freq=160,
                             nb_electrodes=64,
                             nb_trials=45,
                             nb_subjects=20):
    """
    Generate dummy signals with one wavelet per electrode (no mixing sources)
    :param type_data: class of BCI.core.data
    :param start: start of cue in sec
    :param end: end of cue in sec
    :param pattern_len_max: max length of activity inside the activity interval
    :param pattern_len_min: min length of activity inside the activity interval
    :param freq: sampling frequency
    :param nb_electrodes: number of electrodes (one wavelet per electrode)
    :param nb_trials: number of trials to generate per subject
    :param nb_subjects: number of subjects
    :return: array of 'type_data' with a length of 'nb_subjects'
    """

    # generate a noised wavelet signal
    all_data = []
    x_max_len = pattern_len_max * freq
    x_min_len = pattern_len_min * freq
    pattern_x = np.arange(x_max_len)
    data = make_sources(n_s=2 * nb_electrodes, t_samp=pattern_len_max * freq, K=8, w=10)
    data = np.reshape(data, [2, nb_electrodes, -1])
    #np.save(f'./results/source_patterns_{time_str}.npy', data)
    db = 60.0 # SNR in dB
    sig_s = 1.0 # standard deviation of amplitude of sources

    for i in range(nb_subjects):
        labels = np.zeros(shape=(nb_trials,), dtype=np.int32)
        labels[:int(nb_trials / 2)] = 1
        labels = labels[np.random.randn(nb_trials).argsort()]
        length = np.full((nb_trials,), freq, dtype=np.int32)
        start_activity = np.full((nb_trials,), start * freq, dtype=np.int32)
        #length = np.random.randint(x_min_len, x_max_len, (nb_trials,))
        #start_activity = np.random.randint([start * freq] * nb_trials, freq * end - length, (nb_trials,))

        eeg = np.zeros((801, nb_trials, nb_electrodes))

        for j in range(nb_trials):
            for k in range(nb_electrodes):
                sampled_x = np.linspace(0, 1.0, length[j]) * x_max_len
                eeg[start_activity[j]:start_activity[j] + length[j], j, k] = np.interp(sampled_x,
                                                                                       pattern_x,
                                                                                       data[labels[j], k, :])

        eeg = eeg + sig_s * 10 ** (-db / 20) * np.random.randn(*eeg.shape)

        all_data.append(type_data(eeg=eeg,
                                  nb_trials=45,
                                  frequency=freq,
                                  y_dec=labels,
                                  one_hot=None,
                                  y_txt=None,
                                  map_label=None,
                                  chan=np.array([f'fake_{i}' for i in range(nb_electrodes)])))

