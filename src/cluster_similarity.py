import os
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

def load_clusterings(directory, file_pattern="labels_seed_"):
    clusterings = []
    names = []

    for file in sorted(os.listdir(directory)):
        # file is of the form labels_seed_X_0.csv
        if file.startswith(file_pattern):
        
            path = os.path.join(directory, file)
            labels = pd.read_csv(path, header=None).iloc[:,0].values
            clusterings.append(labels)
            names.append(file)

    return clusterings, names


def compute_pairwise_similarity(clusterings):
    n = len(clusterings)
    similarities = []

    for i in range(n):
        for j in range(i+1, n):
            ari = adjusted_rand_score(clusterings[i], clusterings[j])
            similarities.append(ari)
            print(f"Similarity {i}-{j}: {ari:.4f}")

    return np.array(similarities)


def main(directory):

    file_patterns = [
        "labels_seed_", # for GMM clusterings
        "final_consensus_clustering_", # for consensus clusterings
    ]

    for f in file_patterns:

        clusterings, names = load_clusterings(directory, file_pattern=f)

        print(f"Loaded {len(clusterings)} clusterings\n")

        similarities = compute_pairwise_similarity(clusterings)

        print("\nSummary for pattern:", f)
        print("-------")
        print("Average similarity:", similarities.mean())
        print("Std deviation:", similarities.std())


if __name__ == "__main__":
    directory = "output/consensus/gmm_error1.0_scl/small_1000"
    main(directory)