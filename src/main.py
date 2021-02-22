
##### generic library #####
from sklearn.svm import SVC
import scipy.linalg


##### project import #####
from filters.spatial import *
from filters.frequency import *
from utils.read import *


def train_model_SVM_kernel_RBF(cov_train, y_dec):

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
    clf.fit(cov_train, y_dec)
    return clf


def per_subject_classif(all_data):
    for data in all_data:
        clf = train_model_SVM_kernel_RBF(data.features, data.y_dec)


def main():
    all_data = load_mi_classification()
    for data in all_data:
        data.spatial_filter(mi_active_electrodes)
        data.freq_filter(mi_band_pass_filter)
        data.compute_features()

    per_subject_classif(all_data)

if __name__ == "__main__":
    main()
