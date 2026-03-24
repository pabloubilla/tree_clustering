
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from itertools import combinations
from joblib import Parallel, delayed



def main(directory):
    clusters_ward = pd.read_csv(os.path.join(directory, "ward", "final_clusters.csv"), header=None).iloc[:, 0].values
    clusters_average = pd.read_csv(os.path.join(directory, "average", "final_clusters.csv"), header=None).iloc[:, 0].values

    # compute RAND
    rand_index = adjusted_rand_score(clusters_ward, clusters_average)

    print(f"Adjusted Rand Index between Ward and Average linkage: {rand_index:.4f}")

if __name__ == "__main__":
    directory = "output/consensus/gmm_error1.0_scl/full_data"

    # Set saveplot_dir to a path like "plots" to save, or None to skip saving
    main(directory)
    