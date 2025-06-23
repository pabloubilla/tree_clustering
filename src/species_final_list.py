import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """Load necessary datasets from the specified directory."""
    sites_path = os.path.join("data/maps/REV_Community_matrix.csv")
    df_sites = pd.read_csv(sites_path)

    species_path = os.path.join("data/traits_pred_log.csv")
    df_species = pd.read_csv(species_path)
    df_species['accepted_bin'] = df_species['accepted_bin'].str.replace(" ", "_")

    clusters_path = os.path.join(
        "output/consensus_dissertation/gmm_error1.0_scl/full_data/final_clusters.csv"
    )
    df_clusters = pd.read_csv(clusters_path, header=None).to_numpy()

    metrics_path = os.path.join(
        "data/maps/REV_obs_results_200.csv"
    )
    df_metrics = pd.read_csv(metrics_path)

    return df_sites, df_species, df_clusters, df_metrics


def process_data(df_sites, df_species, df_clusters):
    """Merge cluster info into site data and count unique clusters and species per grid."""
    df_species['cluster'] = df_clusters
    df_sites = df_sites.merge(df_species[['accepted_bin', 'cluster']], on='accepted_bin', how='left')

    output_path = os.path.join(
        "output/spatial_analysis"
    )
    os.makedirs(output_path, exist_ok=True)
    # save df_sites to check simpsons calculation
    df_sites.to_csv(os.path.join(output_path,'sites_cluster.csv'))


    # Count unique clusters per grid
    grid_clusters = (
        df_sites.groupby('grid_id')['cluster']
        .nunique()
        .reset_index(name='nclust')
    )

    # Count unique species per grid
    grid_species = (
        df_sites.groupby('grid_id')['accepted_bin']
        .nunique()
        .reset_index(name='nspec')
    )

    # Simpson's index (as simpson measure)
    grid_species['simpson'] = (
        df_sites.groupby('grid_id')['cluster']
        .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
        .reset_index(name='simpson')['simpson']
    )

    # Inverse Simpson
    grid_species['inverse_simpson'] = 1/(1+grid_species['simpson'])


    # average species per cluster
    # grid_species['redun'] = (
    #             df_sites.groupby('grid_id')['cluster']
    #     .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
    #     .reset_index(name='simpson')['simpson']

    # )

    # Merge the two summaries
    grid_summary = pd.merge(grid_clusters, grid_species, on='grid_id')
    

    return df_sites, grid_summary



def plot_cluster_distribution(df_metrics):
    """Plot how many grids have a certain number of clusters."""
    grid_ncluster_count = df_metrics['nclust'].value_counts().reset_index()
    grid_ncluster_count.columns = ['nclust', 'count']
    grid_ncluster_count = grid_ncluster_count.sort_values(by='nclust')

    grid_ncluster_count.plot(kind='bar', x='nclust', y='count')
    plt.xlabel('Number of clusters')
    plt.ylabel('Count')
    plt.title('Number of clusters per grid')
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    plt.show()


def save_output(df_metrics):
    """Save the resulting metrics with cluster info to a CSV file."""
    output_path = os.path.join(
        "output/spatial_analysis"
    )
    os.makedirs(output_path, exist_ok=True)
    df_metrics.to_csv(os.path.join(output_path, 'REV_obs_results_200_extended.csv'), index=False)
    print(f"Saved output to {output_path}")


def main():
    df_sites, df_species, df_clusters, df_metrics = load_data()
    df_sites, grid_clusters = process_data(df_sites, df_species, df_clusters)

    # Merge cluster count into metrics
    df_metrics = df_metrics.merge(grid_clusters, on='grid_id', how='left')

    # Plot distribution
    # plot_cluster_distribution(df_metrics)

    # Save final output
    save_output(df_metrics)


if __name__ == "__main__":
    main()
