
##### generic library #####
from sklearn.svm import SVC
import scipy.linalg

##### eeg library #####
from mne.filter import filter_data

##### project import #####
from filters.spatial import *
from utils.read import *

def main():
    data = load_mi_classification()
    for subject in range(len(data)):
        data[subject] = mi_active_electrodes(data[subject])



if __name__ == "__main__":
    main()
