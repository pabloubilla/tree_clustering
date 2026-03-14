import os
import re
import glob
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def extract_seed(filename):
    """
    Extract from
      labels_seed_12_0.csv
    """
    basename = os.path.basename(filename)

    patterns = [
        r"seed_(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, basename)
        if m:
            return int(m.group(1))

    return None


def select_files_by_seed(directory, seed_min=None, seed_max=None):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    selected = []
    for f in files:
        seed = extract_seed(f)
        if seed is None:
            continue

        if seed_min is not None and seed < seed_min:
            continue
        if seed_max is not None and seed > seed_max:
            continue

        selected.append((seed, f))

    selected.sort(key=lambda x: x[0])
    print(f"Selected {len(selected)} files")

    return selected


def load_labels(files_with_seeds):
    """
    Load all clustering label vectors.
    Assumes all files refer to the same observations in the same order.
    """
    all_labels = []
    seeds = []

    expected_n = None

    for seed, f in files_with_seeds:
        labels = pd.read_csv(f, header=None).iloc[:, 0].to_numpy()

        if expected_n is None:
            expected_n = len(labels)
        elif len(labels) != expected_n:
            raise ValueError(
                f"File {f} has {len(labels)} rows, expected {expected_n}."
            )

        all_labels.append(labels)
        seeds.append(seed)

    return np.array(all_labels), seeds


def build_consensus_matrix(label_runs, ignore_label=-1):
    """
    Consensus(i,j) = fraction of selected runs in which i and j
    are assigned to the same cluster.

    If ignore_label is not None, observations with that label
    are excluded from the denominator for that run.
    """
    n_runs, n_obs = label_runs.shape

    same_counts = np.zeros((n_obs, n_obs), dtype=float)
    valid_counts = np.zeros((n_obs, n_obs), dtype=float)

    for labels in label_runs:
        if ignore_label is None:
            valid = np.ones(n_obs, dtype=bool)
        else:
            valid = labels != ignore_label

        valid_pair = np.outer(valid, valid)
        same_pair = (labels[:, None] == labels[None, :]) & valid_pair

        same_counts += same_pair
        valid_counts += valid_pair

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(
            same_counts,
            valid_counts,
            out=np.zeros_like(same_counts),
            where=valid_counts > 0
        )

    np.fill_diagonal(consensus, 1.0)
    return consensus


def consensus_to_distance(consensus):
    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)
    return distance


def hierarchical_labels(distance_matrix, n_clusters, linkage_method="average"):
    """
    Fit hierarchical clustering from a precomputed distance matrix.
    For consensus distances, 'average' is usually a safer choice than 'ward'.
    """
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = cut_tree(Z, n_clusters=[n_clusters]).reshape(-1)
    return labels


def find_best_k(distance_matrix, k_values, linkage_method="average"):
    scores = []

    for k in k_values:
        labels = hierarchical_labels(distance_matrix, k, linkage_method=linkage_method)
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        scores.append(score)
        print(f"k={k:3d} silhouette={score:.4f}")

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_k = k_values[best_idx]
    best_score = scores[best_idx]

    return best_k, best_score, scores


def main():
    # ---------------------------
    # EDIT THESE FOR YOUR USE CASE
    # ---------------------------
    input_dir = "output/consensus/gmm_error1.0_scl/small_1000"

    # seed_min_list = [1,51,101,151,201]
    # seed_max_list    = [50,100,150,200,250]

    size_per_seed = 50
    seed_min_list = [1, 51, 101, 151]
    seed_max_list = [s + size_per_seed - 1 for s in seed_min_list]



    k_min = 2
    k_max = 10

    linkage_method = "ward"   # recommended for a distance matrix like 1 - consensus
    ignore_label = -1            # set to None if you do not want special handling for -1
    # output_file = "final_consensus_clustering.csv"
    # ---------------------------

    final_labels_dict = {}

    for idx, (seed_min, seed_max) in enumerate(zip(seed_min_list, seed_max_list)):
        print(f"\n=== Processing seeds {seed_min} to {seed_max} ===")

        files_with_seeds = select_files_by_seed(input_dir, seed_min=seed_min, seed_max=seed_max)

        if len(files_with_seeds) < 2:
            raise ValueError("Need at least 2 clustering files in the selected seed range.")

        print("Selected files:")
        for seed, f in files_with_seeds:
            print(f"  seed={seed:3d}  {os.path.basename(f)}")

        label_runs, seeds = load_labels(files_with_seeds)
        print(f"\nLoaded {label_runs.shape[0]} runs and {label_runs.shape[1]} observations.")

        consensus = build_consensus_matrix(label_runs, ignore_label=ignore_label)
        distance = consensus_to_distance(consensus)

        k_values = list(range(k_min, k_max + 1))
        best_k, best_score, scores = find_best_k(distance, k_values, linkage_method=linkage_method)

        print(f"\nBest k: {best_k}")
        print(f"Best silhouette: {best_score:.4f}")

        final_labels = hierarchical_labels(distance, best_k, linkage_method=linkage_method)

        # final_labels_dict[idx] = final_labels

        out_path = os.path.join(input_dir, f"final_consensus_clustering_{seed_min}_{seed_max}.csv")
        pd.DataFrame(final_labels).to_csv(out_path, header=False, index=False)

        # out_path = os.path.join(input_dir, output_file)

        # print(f"Saved final clustering to: {out_path}")


if __name__ == "__main__":
    main()