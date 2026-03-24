import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from itertools import combinations
from joblib import Parallel, delayed




# def _pairwise_ari(i, j, clusterings):
#     ari = adjusted_rand_score(clusterings[i], clusterings[j])
#     return i, j, ari


# def compute_pairwise_similarity(clusterings, n_jobs=-1, verbose=True):
#     n = len(clusterings)
#     pairs = list(combinations(range(n), 2))

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(_pairwise_ari)(i, j, clusterings)
#         for i, j in pairs
#     )

#     similarities = []
#     for i, j, ari in results:
#         similarities.append(ari)
#         if verbose:
#             print(f"Similarity {i}-{j}: {ari:.4f}")

#     return np.array(similarities)



def load_clusterings(directory, file_pattern="labels_seed_"):
    clusterings = []
    names = []

    for file in sorted(os.listdir(directory)):
        # file is of the form labels_seed_X_0.csv
        if file.startswith(file_pattern):
            path = os.path.join(directory, file)
            labels = pd.read_csv(path, header=None).iloc[:, 0].values
            clusterings.append(labels)
            names.append(file)

    return clusterings, names


def compute_pairwise_similarity(clusterings):
    n = len(clusterings)
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(clusterings[i], clusterings[j])
            similarities.append(ari)
            print(f"Similarity {i}-{j}: {ari:.4f}")

    return np.array(similarities)


def plot_similarity_density(

    similarities_by_pattern,
    show_plot=True,
    saveplot_dir=None,
    filename="pairwise_similarity_density.png"
):

    pattern_mapping = {
        "labels_seed_": "GMM Clusters",
        "final_consensus_clustering_": "Consensus Clusters",
    }

    plt.figure(figsize=(10, 6))

    for pattern, similarities in similarities_by_pattern.items():
        label = pattern_mapping.get(pattern, pattern)
        if len(similarities) < 2:
            print(f"Skipping '{pattern}' (not enough samples for KDE)")
            continue

        sns.kdeplot(
            similarities,
            label=label,
            fill=True,
            common_norm=False,   # keeps densities independent
            alpha=0.3,
            clip=(-1, 1)
        )

    plt.xlabel("Adjusted Rand Index (ARI)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if saveplot_dir is not None:
        os.makedirs(saveplot_dir, exist_ok=True)
        save_path = os.path.join(saveplot_dir, 'images', filename)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()



def main(directory, show_plot=True, saveplot_dir=None):
    file_patterns = [
        "labels_seed_",                 # for GMM clusterings
        "final_consensus_clustering_",  # for consensus clusterings
    ]

    similarities_by_pattern = {}

    for f in file_patterns:
        clusterings, names = load_clusterings(directory, file_pattern=f)

        print(f"Pattern: {f}")
        print(f"Loaded {len(clusterings)} clusterings\n")

        if len(clusterings) < 2:
            print(f"Not enough clusterings for pattern '{f}' to compute pairwise similarities.\n")
            similarities = np.array([])
        else:
            similarities = compute_pairwise_similarity(clusterings)

            print("\nSummary for pattern:", f)
            print("-------")
            print("Average similarity:", similarities.mean())
            print("Std deviation:", similarities.std())
            print()

        similarities_by_pattern[f] = similarities

    plot_similarity_density(
        similarities_by_pattern,
        show_plot=show_plot,
        saveplot_dir=saveplot_dir,
    )


if __name__ == "__main__":
    directory = "output/consensus/gmm_error1.0_scl/small_0"

    # Set saveplot_dir to a path like "plots" to save, or None to skip saving
    main(directory, show_plot=True, saveplot_dir=directory)
    