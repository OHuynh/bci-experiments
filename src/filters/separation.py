##### generic library #####
import numpy as np
from sklearn.decomposition import FastICA

##### BSS library import #####
from pyDecGMCA.algoDecG import *
from pyDecGMCA.mathTools import *

from smica.core_smica import SMICA


def fastICA(data, n_components=4):
    eeg = data.eeg
    transformer = FastICA(n_components=n_components, random_state=0, max_iter=1000)
    sources = np.zeros(shape=(eeg.shape[0], eeg.shape[1], n_components))
    for trial in range(eeg.shape[1]):
        sources[:, trial, :] = transformer.fit_transform(eeg[:, trial, :])
    return sources


def decGMCA(data, n_components=5):
    """
    joint deconvolution and blind source separation from [1]

    [1] : Joint Multichannel Deconvolution and Blind Source Separation. Ming Jiang, Jerome Bobin, Jean-Luc Starck.
          SIAM Journal on Imaging Sciences, Society for Industrial and Applied Mathematics, 2017
    :param data:
    :return:
    """
    eeg = data.eeg
    Nx = 1
    Ny = eeg.shape[0]
    Imax = 100
    epsilon = np.array([1e0])  # Tikhonov parameter
    epsilonF = 1e-5
    Ndim = 1 # 1D Signal
    fc = 1. / 32
    trials = eeg.shape[1]
    sources = np.zeros(shape=(eeg.shape[0], trials, n_components))
    for trial in range(trials):
        from sklearn.preprocessing import normalize
        V_N = fftNd1d(normalize(eeg[:, trial, :]).transpose(), 1)
        kernMat = np.ones((eeg.shape[2], eeg.shape[0]), dtype='complex128')
        (S_est,A_est) = DecGMCA(V_N, kernMat, n_components, Nx, Ny, Imax, Ndim, epsilonF, Ndim, wavelet=True, scale=4,
                                mask=False, deconv=True,wname='starlet', thresStrtg=2,
                                FTPlane=True, fc=fc, logistic=True, postProc=True, Ksig=0.6)
        sources[:, trial, :] = S_est.transpose()
        #print(S_est)
    return sources

def smica(data, freqs,  n_components=20, **kwargs):
    """
    Spectral Matching Independent Component Analysis [1]

    :param data:
    :param n_components: 
    :return: 
    """
    X = data.eeg.transpose([2, 1, 0])
    X = X.reshape([X.shape[0], -1])
    normalization = np.std(X)
    X /= normalization

    scaling = np.std(X, axis=1)
    X /= scaling[:, None]

    smica = SMICA(n_components, freqs, data.sampling_frequency, avg_noise=False, corr=False)
    smica.fit(X, **kwargs)
    sources = smica.compute_sources().reshape(n_components, data._nb_trials, -1)
    return sources.transpose([2, 1, 0])
