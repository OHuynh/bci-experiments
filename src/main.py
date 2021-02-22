
##### generic library #####
from sklearn.svm import SVC
import scipy.linalg


##### project import #####
from filters.spatial import *
from filters.frequency import *
from utils.read import *

def main():
    all_data = load_mi_classification()
    for data in all_data:
        data.spatial_filter(mi_active_electrodes)
        data.freq_filter(mi_band_pass_filter)

        data.compute_features()

if __name__ == "__main__":
    main()
