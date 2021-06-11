##### generic library #####
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import scipy.linalg


##### project import #####
from filters.spatial import *
from filters.frequency import *
from filters.decomposition import *
from core.data import *
from utils.read import *

def estimator_SVM_kernel_RBF():

    def log_euclidean_gaussian_rbf(X, Y):
        """
        See paper [1] for a list of compatible distances to get rbf kernel positive definite

        [1] : Kernel Methods on Riemannian Manifolds with Gaussian RBF Kernels, S. Jayasumana, R. Hartley,
              M. Salzmann, H. Li and M. Harandi, PAMI, 2015
        :param X: cov matrice 1
        :param Y: cov matrice 2
        :return: kernel between two covariance matrices
        """
        distances = np.zeros(shape=(X.shape[0], Y.shape[0]))
        dim = int(np.sqrt(X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                #log euclidean distance
                distances[i, j] = np.linalg.norm(scipy.linalg.logm(np.reshape(X[i], (dim, dim)))
                                                 - scipy.linalg.logm(np.reshape(Y[j], (dim, dim))),
                                                 ord='fro')
        return np.exp(- 1.0 * distances ** 2)

    #kernelized svm with RBF
    clf = SVC(C=0.1, kernel=log_euclidean_gaussian_rbf)
    return clf

def within_subject_classif(all_data_train, all_data_test):
    cv = KFold(n_splits=10, shuffle=True)
    for subject in range(len(all_data_train)):
        features = np.concatenate([all_data_train[subject].features, all_data_test[subject].features]
                                  , axis=0)
        labels = np.concatenate([all_data_train[subject].y_dec, all_data_test[subject].y_dec])
        clf = estimator_SVM_kernel_RBF()
        scores = cross_val_score(clf, features, labels, cv=cv, n_jobs=1)
        print("RBF SVM Classification accuracy: {} ".format(np.mean(scores)))

def process_data(all_data):
    for data in all_data:
        data.spatial_filter(mi_active_electrodes)
        data.freq_filter(mi_band_pass_filter)
        data.decomposition(decGMCA)
        data.plot_eeg(label=1, mode='raw')

        #data.window_crop(3000, 4000)

        data.compute_features()
    return all_data


def main():
    all_data_train = load_osf_mi_classification(TimeFrequencyData, 'Training Set')
    all_data_test = load_osf_mi_classification(TimeFrequencyData, 'Test Set')

    #all_data_train = load_eegbci_mi_classification(TimeFrequencyData)

    all_data_train = process_data(all_data_train)
    #all_data_test = process_data(all_data_test)+

    within_subject_classif(all_data_train, all_data_test)

if __name__ == "__main__":
    main()
