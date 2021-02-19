import numpy as np

def mi_active_electrodes(data):
    """
    This function filters electrode close to motor cortex location involved in MI activity according to [1] :
    F3,Fz,F4,FC1,FC2,C3,Cz,C4,CP1,CP2,P3,P4

    [1] Exploring invariances of multivariate time series via Riemannian geometry: validation on EEG data
        , 2019, Pedro Luiz Coelho Rodrigues (page 55)

    :param data: array of data tuples from osf 2020 bci competition
    :return: tuple with filtered data
    """
    list_electrodes = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'P4']
    list_electrodes = np.array(list_electrodes)
    indices_to_keep = np.where((data.chan == list_electrodes.reshape(-1, 1)))[1]
    indices = np.zeros(62, dtype=np.bool)
    indices[indices_to_keep] = True

    data.eeg = data.eeg[:, :, indices]
    data.chan = list_electrodes

    return data