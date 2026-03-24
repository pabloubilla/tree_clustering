import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  # 10pt font size for text elements
LINEWIDTH = 6.30045
from scipy.stats import spearmanr

def spearman_summary(df, pairs, nice_names=None, out_csv=None):
    """
    Compute Spearman rho (and p) for a list of (y, x) pairs.
    Returns a DataFrame with columns: y, x, rho, p, n, pair.
    """
    rows = []
    for y, x in pairs:
        sub = df[[x, y]].dropna()
        n = len(sub)
        if n < 2:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(sub[x], sub[y])
        rows.append({
            "y": y, "x": x,
            "rho": rho, "p": p, "n": n,
            "pair": f"{(nice_names or {}).get(y, y)} vs {(nice_names or {}).get(x, x)}"
        })
    res = pd.DataFrame(rows).sort_values(["y", "x"])
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        res.to_csv(out_csv, index=False)
    return res


def plot_metric(df, x, y, path):

    plt.figure(figsize=(5,5))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.2, s=3, c='darkblue')
    # plt.show()
    plt.ylim(df[y].min()*0.95,df[y].max()*1.05)
    plt.savefig(path)
    plt.close()


def plot_metric_grid(df, x_list, y_list, xlabel_list, ylabel_list, output_path):
    n_rows = len(y_list)
    n_cols = len(x_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(LINEWIDTH, 4.5), sharey='row', sharex='col')
    fig.subplots_adjust(hspace=.2, wspace=.2)

    # Generate subplot labels: a), b), c)...
    labels = [f"{chr(97+k)})" for k in range(n_rows * n_cols)]

    for i, y in enumerate(y_list):
        for j, x in enumerate(x_list):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            sns.scatterplot(data=df, x=x, y=y, alpha=0.2, s=1.5, color='darkblue', ax=ax)

            # Add subplot label in bold at top middle
            label_idx = i * n_cols + j
            ax.text(
                0.5, 1.02,  # middle top
                labels[label_idx],
                transform=ax.transAxes,
                fontweight='bold',
                fontsize=10,
                va='bottom',
                ha='center'
            )

            if j == 0:
                ax.set_ylabel(ylabel_list[i])
            else:
                ax.set_ylabel('')

            if i == n_rows - 1:
                ax.set_xlabel(xlabel_list[j])
            else:
                ax.set_xlabel('')

    fig.savefig(output_path + '.png', bbox_inches='tight', pad_inches=0.2, dpi=1200)
    fig.savefig(output_path + '.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()


if __name__ == '__main__':
    # Load metrics CSV
    metrics_path = "output/spatial_analysis/REV_obs_results_200_extended.csv"
    fig_path = 'output/figures/metrics'
    df = pd.read_csv(metrics_path)

    # at least 3 clusters
    df = df[df['nclust'] > 3]

    print(len(df))

    x_list = ['nspec', 'raoq', 'fdr']
    y_list = ['nclust', 'cluster_simpson']

    xlabel_list = ['Species Richness', 'Rao\'s Q \n (mean pairwise distance)', 'Functional Richness \n (convex hull)']
    ylabel_list = ['Functional Group Richess', 'Functional Redundancy \n (Simpson\'s Index)']

    os.makedirs(fig_path, exist_ok=True)

    for x in x_list:
        for y in y_list:
            plot_metric(df, x, y, os.path.join(fig_path,f'{x}_{y}.pdf'))


    plot_metric_grid(df, x_list, y_list, xlabel_list, ylabel_list, os.path.join(fig_path, 'metric_grid'))


    nice = {
        "nclust": "FGR",
        "cluster_simpson": "FRedund",
        "nspec": "Species richness",
        "raoq": "Rao's Q",
        "fdr": "Functional richness"
    }

    pairs = [
        ("nclust", "nspec"),             # FGR ~ species richness
        ("cluster_simpson", "nspec"),    # FRedund ~ species richness
        ("nclust", "raoq"),              # FGR ~ Rao's Q
        ("cluster_simpson", "raoq"),     # FRedund ~ Rao's Q
        ("nclust", "fdr"),               # FGR ~ functional richness
        ("cluster_simpson", "fdr")       # FRedund ~ functional richness
    ]

    corr_tbl = spearman_summary(df, pairs, nice_names=nice,
                                out_csv="output/spatial_analysis/spearman_correlations.csv")
    print(corr_tbl)