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


def generate_wavelet_sources(pattern_len, nb_electrodes, type_data):
    # generate a noised wavelet
    all_data = []
    nb_electrodes = 1
    pattern_len = 380
    pattern_x = np.arange(pattern_len)
    data = make_sources(n_s=2 * nb_electrodes, t_samp=380, K=8, w=10)
    data = np.reshape(data, [2, nb_electrodes, -1])
    #np.save(f'./results/source_patterns_{time_str}.npy', data)
    np.load('./results/source_patterns_2022_07_25_02_40.npy')
    db = 60.0 # SNR in dB
    sig_s = 1.0 # standard deviation of amplitude of sources

    for i in range(20):
        nb_trials = 45
        labels = np.zeros(shape=(nb_trials,), dtype=np.int32)
        labels[:int(nb_trials / 2)] = 1
        labels = labels[np.random.randn(nb_trials).argsort()]
        length = np.full((nb_trials,), 160, dtype=np.int32)
        start_activity = np.full((nb_trials,), 180, dtype=np.int32)
        #length = np.random.randint(160 * 0.5, 160 * 2.0, (nb_trials,))
        #start_activity = np.random.randint([160] * nb_trials, 160.0 * 5.0 - length, (nb_trials,))

        eeg = np.zeros((801, nb_trials, nb_electrodes))

        for j in  range(nb_trials):
            for k in range(nb_electrodes):
                eeg[start_activity[j]:start_activity[j] + length[j], j, k] = np.interp(np.linspace(0, 1.0, length[j]) * pattern_len, pattern_x, data[labels[j], k, :])

        eeg = eeg + sig_s * 10 ** (-db / 20) * np.random.randn(*eeg.shape)

        all_data.append(type_data(eeg=eeg,
                                       nb_trials=45,
                                       frequency=160,
                                       y_dec=labels,
                                       one_hot=None,
                                       y_txt=None,
                                       map_label=None,
                                       chan=np.array([f'fake_{i}' for i in range(nb_electrodes)])))

