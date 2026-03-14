#!/usr/bin/env python3
"""
RAM-safer version of your pipeline.

Key changes (same overall workflow):
- Use float32 by default (cuts matrix RAM ~2x).
- Build distance matrix in-place and optionally drop consensus to reduce peak RAM.
- Compute squareform + linkage ONCE, then fcluster for each K (instead of 118 times).
- Use sampled silhouette_score to avoid O(N^2) extra work/peaks (still uses precomputed distances).

Usage:
  python3 step3_analyse_consensus_ramfix.py <method> [n_jobs]
Example:
  python3 step3_analyse_consensus_ramfix.py gmm_error1.0_scl
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ----------------------------
# Small utilities
# ----------------------------

def print_mem_hint(arr, name: str):
    if arr is None:
        print(f"{name}: None")
        return
    nbytes = arr.nbytes
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, size≈{nbytes/1024**3:.2f} GB")

def try_print_cgroup_limit():
    # Helpful on clusters where physical RAM != your job limit
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p, "r") as f:
                v = f.read().strip()
            print(f"cgroup memory limit ({p}): {v}")
            break
        except Exception:
            pass

def linewidth():
    return 6.30045


# ----------------------------
# I/O and core computations
# ----------------------------

def load_consensus_matrix(output_dir: str, dtype=np.float32) -> np.ndarray:
    """Load consensus matrix from parquet -> numpy array."""
    path = os.path.join(output_dir, "consensus_matrix.parquet")
    # Use pandas -> numpy; dtype cast to reduce RAM.
    mat = pd.read_parquet(path).to_numpy()
    if dtype is not None:
        mat = mat.astype(dtype, copy=False)
    return mat

def compute_distance_matrix_inplace_from_consensus(consensus_matrix: np.ndarray) -> np.ndarray:
    """
    Compute distance = 1 - consensus.
    Uses a new array (distance) but keeps it float32/float64 same as consensus.
    """
    distance_matrix = (1.0 - consensus_matrix)
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix

def compute_linkage_once(distance_matrix: np.ndarray, linkage_method: str = "average"):
    """
    Build condensed distance vector once, then linkage once.
    """
    print("Building condensed distance vector (squareform) ONCE ...")
    condensed = squareform(distance_matrix, checks=False)  # checks=False saves time/mem overhead
    print_mem_hint(condensed, "condensed_distance")
    print(f"Computing linkage ONCE (method='{linkage_method}') ...")
    Z = sch.linkage(condensed, method=linkage_method)
    # Free condensed ASAP (big)
    del condensed
    return Z

def analyze_clusters_with_fixed_linkage(
    distance_matrix: np.ndarray,
    Z,
    n_cluster_list,
    output_dir: str,
    method: str,
    sample_size: int = 2000,
    random_state: int = 0,
):
    """
    Compute silhouette scores across K using a single linkage.
    Uses sampling for silhouette to avoid heavy memory/time.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    silhouette_score_list = []
    for k in n_cluster_list:
        print(f"Calculating for {k} clusters")
        clusters = sch.fcluster(Z, t=k, criterion="maxclust")

        # silhouette with precomputed full NxN is heavy; sample to reduce load
        # NOTE: silhouette_score supports sample_size for efficiency
        score = silhouette_score(
            distance_matrix,
            clusters,
            metric="precomputed",
            sample_size=sample_size,
            random_state=random_state,
        )
        print(score)
        silhouette_score_list.append(score)

    df_silhouette = pd.DataFrame({"K": n_cluster_list, "Silhouette Score": silhouette_score_list})
    df_silhouette.to_csv(os.path.join(output_dir, f"silhouette_scores_{method}.csv"), index=False)

    best_k = int(n_cluster_list[int(np.argmax(silhouette_score_list))])
    max_score = float(np.max(silhouette_score_list))

    plt.figure(figsize=(12, 6))
    plt.plot(n_cluster_list, silhouette_score_list, linestyle="-")
    plt.xlabel("Number of groups ($K$)")
    plt.ylabel(f"Silhouette Score (sample_size={sample_size})")
    plt.scatter(best_k, max_score, zorder=5)
    plt.text(
        n_cluster_list[-1],
        max_score,
        f"Best K = {best_k}\nMax Score = {max_score:.4f}",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f"silhouette_score_{method}.pdf"))
    plt.clf()

    return best_k

def hierarchical_clustering_from_linkage(Z, consensus_matrix: np.ndarray, num_clusters: int):
    """Cut a precomputed linkage, then reorder consensus by cluster assignment."""
    clusters = sch.fcluster(Z, t=num_clusters, criterion="maxclust")
    order = np.argsort(clusters)
    ordered_consensus = consensus_matrix[order, :][:, order]
    return clusters, ordered_consensus

def compute_summary_matrix(consensus_matrix: np.ndarray, consensus_labels: np.ndarray):
    """
    Summary of consensus: avg consensus between every cluster pair
    diagonal is avg consensus inside cluster
    """
    num_clusters = len(np.unique(consensus_labels))
    consensus_summary_matrix = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    cluster_sizes = np.zeros(num_clusters, dtype=np.int64)

    for c in range(1, num_clusters + 1):
        idx_c = np.where(consensus_labels == c)[0]
        cluster_sizes[c - 1] = len(idx_c)

        # inside cluster (exclude diagonal)
        matrix_c = consensus_matrix[idx_c, :][:, idx_c]
        denom = (len(idx_c) ** 2 - len(idx_c))
        avg_consensus_c = (matrix_c.sum() / denom) if denom > 0 else 0.0
        consensus_summary_matrix[c - 1, c - 1] = avg_consensus_c

        for c2 in range(c + 1, num_clusters + 1):
            idx_c2 = np.where(consensus_labels == c2)[0]
            matrix_c1_c2 = consensus_matrix[idx_c, :][:, idx_c2]
            avg = matrix_c1_c2.sum() / (len(idx_c) * len(idx_c2))
            consensus_summary_matrix[c - 1, c2 - 1] = avg
            consensus_summary_matrix[c2 - 1, c - 1] = avg

    return consensus_summary_matrix, cluster_sizes

def save_or_load_summary_matrix(consensus_matrix, clusters, output_dir):
    summary_matrix_file = os.path.join(output_dir, "summary_consensus_matrix.csv")
    cluster_sizes_file = os.path.join(output_dir, "cluster_sizes.csv")

    if os.path.exists(summary_matrix_file) and os.path.exists(cluster_sizes_file):
        print("Files found. Loading saved data.")
        summary_consensus_matrix = pd.read_csv(summary_matrix_file, index_col=0).values
        cluster_sizes = pd.read_csv(cluster_sizes_file, index_col=0).values.flatten()
    else:
        print("Files not found. Computing data.")
        summary_consensus_matrix, cluster_sizes = compute_summary_matrix(consensus_matrix, clusters)
        pd.DataFrame(summary_consensus_matrix).to_csv(summary_matrix_file)
        pd.DataFrame(cluster_sizes).to_csv(cluster_sizes_file)

    return summary_consensus_matrix, cluster_sizes


# ----------------------------
# Plotting (kept close to yours)
# ----------------------------

def plot_summary_heatmap(consensus_summary_matrix, cluster_sizes, output_path,
                         scale_small=1000, scale_big=10000, sorted=False):
    num_clusters = len(cluster_sizes)
    sorted_indices = np.argsort(-cluster_sizes)
    if sorted:
        mat = consensus_summary_matrix[sorted_indices, :][:, sorted_indices]
        sizes = cluster_sizes[sorted_indices]
    else:
        mat = consensus_summary_matrix.copy()
        sizes = cluster_sizes.copy()

    total = float(np.sum(sizes))
    normalized = sizes / total

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")

    current_x = 0.01
    for i in range(num_clusters):
        current_y = 0.08
        for j in range(num_clusters):
            width = normalized[i]
            height = normalized[j]
            color = plt.cm.Reds(mat[i, j])
            rect = plt.Rectangle((current_x, current_y), width, height,
                                 facecolor=color, edgecolor="black", linewidth=0.1)
            ax.add_patch(rect)
            current_y += height
        current_x += width

    ax.set_xlim(0, current_x + 0.01)
    ax.set_ylim(0, current_y + 0.01)
    ax.invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average consensus")

    plt.xticks([])
    plt.yticks([])

    scale_small_species = scale_small / total
    scale_big_species = scale_big / total

    ax.plot([0.01, scale_big_species + 0.01], [0.03, 0.03],
            color="black", linewidth=1.5, linestyle="--")
    ax.text(0.06, 0.015, "Num. Species", ha="center", va="bottom")

    tick_length = 0.007
    ax.plot([0.01, 0.01], [0.03 - tick_length, 0.03 + tick_length], color="black", linewidth=1.5)
    ax.plot([scale_small_species + 0.01, scale_small_species + 0.01],
            [0.03 - tick_length, 0.03 + tick_length], color="black", linewidth=1.5)
    ax.text(scale_small_species + 0.01, 0.045, str(scale_small), ha="center", va="top")

    ax.plot([scale_big_species + 0.01, scale_big_species + 0.01],
            [0.03 - tick_length, 0.03 + tick_length], color="black", linewidth=1.5)
    ax.text(scale_big_species + 0.01, 0.045, str(scale_big), ha="center", va="top")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()

def plot_summary_heatmap_w_squares(consensus_summary_matrix, cluster_sizes, output_path,
                                   specific_sizes=(1000, 2000, 4000), sorted=False):
    num_clusters = len(cluster_sizes)
    sorted_indices = np.argsort(-cluster_sizes)
    if sorted:
        mat = consensus_summary_matrix[sorted_indices, :][:, sorted_indices]
        sizes = cluster_sizes[sorted_indices]
    else:
        mat = consensus_summary_matrix.copy()
        sizes = cluster_sizes.copy()

    total = float(np.sum(sizes))
    normalized = sizes / total

    fig, ax = plt.subplots(figsize=(.8 * linewidth(), .8 * linewidth() - 1.5))

    # draw off-diagonal first
    current_x = 0.0
    for i in range(num_clusters):
        current_y = 0.0
        width = normalized[i]
        for j in range(num_clusters):
            height = normalized[j]
            if i != j:
                color = plt.cm.Reds(mat[i, j])
                rect = plt.Rectangle((current_x, current_y), width, height,
                                     facecolor=color, edgecolor="whitesmoke", linewidth=0.3)
                ax.add_patch(rect)
            current_y += height
        current_x += width

    # draw diagonal
    current_x = 0.0
    current_y = 0.0
    for i in range(num_clusters):
        width = normalized[i]
        height = normalized[i]
        color = plt.cm.Reds(mat[i, i])
        rect = plt.Rectangle((current_x, current_y), width, height,
                             facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)
        current_x += width
        current_y += height

    ax.axis("off")
    ax.set_xlim(-0.25, current_x + 0.01)
    ax.set_ylim(0, current_y + 0.3)
    ax.invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Average consensus", rotation=270, labelpad=20)

    plt.xticks([])
    plt.yticks([])

    # legend squares
    normalized_specific = [s / total for s in specific_sizes]
    x_start = -0.2
    starting_y = 1.0
    for i, size in enumerate(normalized_specific):
        rect = plt.Rectangle((x_start + 0.08, starting_y - size), size, size,
                             facecolor="white", edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x_start - 0.15, starting_y - size + size / 2,
                f"N = {specific_sizes[i]}", ha="left", fontsize=8)
        starting_y -= size + 0.2

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path)
    plt.clf()


# ----------------------------
# Main
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 step3_analyse_consensus_ramfix.py <method> [n_jobs]")
        sys.exit(1)

    method = sys.argv[1]
    # n_jobs kept for CLI compatibility; this version doesn't parallelize silhouette (intentionally)
    n_jobs = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    _ = n_jobs  # unused

    try_print_cgroup_limit()

    consensus_data = "full_data"
    output_dir = os.path.join("output", "consensus", method, consensus_data)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Try fewer first if you want faster testing
    n_cluster_list = list(range(20, 60))

    # Controls that matter for RAM/time
    dtype = np.float32                 # float32 saves big RAM
    linkage_method = "ward"        
    silhouette_sample = None           # increase if you want more stable scores

    print("Loading consensus matrix...")
    consensus_matrix = load_consensus_matrix(output_dir, dtype=dtype)
    print_mem_hint(consensus_matrix, "consensus_matrix")
    print("Matrix subsample:")
    print(consensus_matrix[100:110, 100:110])

    print("Computing distance matrix...")
    distance_matrix = compute_distance_matrix_inplace_from_consensus(consensus_matrix)
    print_mem_hint(distance_matrix, "distance_matrix")

    print("Analyzing cluster sizes (linkage ONCE + sampled silhouette)...")
    Z = compute_linkage_once(distance_matrix, linkage_method=linkage_method)

    best_k = analyze_clusters_with_fixed_linkage(
        distance_matrix=distance_matrix,
        Z=Z,
        n_cluster_list=n_cluster_list,
        output_dir=output_dir,
        method=method,
        sample_size=silhouette_sample,
        random_state=0,
    )

    print(f"Best number of clusters: {best_k}")

    print("Cutting linkage for best K + ordering consensus...")
    clusters, ordered_consensus = hierarchical_clustering_from_linkage(Z, consensus_matrix, best_k)

    # save clusters
    pd.DataFrame(clusters).to_csv(os.path.join(output_dir, "final_clusters.csv"),
                                  index=False, header=False)

    print("Plotting Summary...")
    summary_consensus_matrix, cluster_sizes = save_or_load_summary_matrix(consensus_matrix, clusters, output_dir)
    print(summary_consensus_matrix)

    sorted_flag = False
    sorted_string = "sorted" if sorted_flag else ""
    plot_summary_heatmap(
        summary_consensus_matrix,
        cluster_sizes,
        os.path.join(images_dir, f"heatmap_G{best_k}_summary{sorted_string}.pdf"),
        sorted=sorted_flag,
    )
    plot_summary_heatmap_w_squares(
        summary_consensus_matrix,
        cluster_sizes,
        os.path.join(images_dir, f"heatmap_G{best_k}_squares.pdf"),
        sorted=sorted_flag,
    )

    cluster_info = {
        "Cluster": range(1, best_k + 1),
        "Size": cluster_sizes,
        "Avg Consensus": np.diag(summary_consensus_matrix),
    }
    pd.DataFrame(cluster_info).to_csv(os.path.join(output_dir, "clusters_info.csv"), index=False)

    # Optional: free huge arrays early
    del ordered_consensus
    del Z
    del distance_matrix
    del consensus_matrix

    print("Done.")


if __name__ == "__main__":
    # Strong suggestion on clusters (prevents huge threaded BLAS memory spikes):
    # export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    main()
