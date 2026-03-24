import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    input_path = 'output/consensus/gmm_error1.0_scl/full_data'
    parser.add_argument("--output_csv", default="species_stability.csv")
    # parser.add_argument("--output_hist", default="species_stability_hist.pdf")
    parser.add_argument("--exclude_self", action="store_true", help="Exclude Q_ii from the sums")
    args = parser.parse_args()

    consensus_matrix_path = os.path.join(input_path, "consensus_matrix.parquet")
    Q = pd.read_parquet(consensus_matrix_path).to_numpy(dtype=np.float32)
    clusters = pd.read_csv(os.path.join(input_path, "ward", "final_clusters.csv"), header=None).iloc[:, 0].to_numpy()

    n = Q.shape[0]
    stability = np.zeros(n, dtype=np.float32)

    for i in range(n):
        same_cluster = clusters == clusters[i]

        if args.exclude_self:
            same_cluster[i] = False
            denom_mask = np.ones(n, dtype=bool)
            denom_mask[i] = False
        else:
            denom_mask = np.ones(n, dtype=bool)

        num = Q[i, same_cluster].sum()
        den = Q[i, denom_mask].sum()
        stability[i] = num / den if den > 0 else np.nan

    df = pd.DataFrame({
        "species_index": np.arange(n),
        "cluster": clusters,
        "stability": stability
    })
    df.to_csv(args.output_csv, index=False)

    plt.figure(figsize=(5, 3.5))
    plt.hist(stability[~np.isnan(stability)], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Species stability")
    plt.ylabel("Count")
    plt.tight_layout()
    # print std and mean
    print(f"Mean stability: {np.nanmean(stability):.4f}")
    print(f"Std stability: {np.nanstd(stability):.4f}")

    image_path = os.path.join(input_path, 'ward', 'images', "species_stability_hist.png")
    plt.savefig(image_path, dpi=300)


if __name__ == "__main__":
    main()