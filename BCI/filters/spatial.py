import numpy as np

def mi_active_electrodes(eeg, chan):
    """
    This function filters electrode close to motor cortex location involved in MI activity according to [1] :
    F3,Fz,F4,FC1,FC2,C3,Cz,C4,CP1,CP2,P3,P4

    [1] Exploring invariances of multivariate time series via Riemannian geometry: validation on EEG data
        , 2019, Pedro Luiz Coelho Rodrigues (page 55)

    :param data: array of data tuples from osf 2020 bci competition
    :return: tuple with filtered data
    """
    list_electrodes = ['F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'P4']
    list_electrodes = np.array(list_electrodes)
    indices_to_keep = np.where((chan == list_electrodes.reshape(-1, 1)))[1]
    indices = np.zeros(chan.shape[0], dtype=np.bool)
    indices[indices_to_keep] = True

    eeg = eeg[:, :, indices]
    chan = list_electrodes

    return eeg, chan