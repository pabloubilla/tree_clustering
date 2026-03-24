import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """Load community matrix, species traits, clustering results, taxonomic info, and grid coordinates."""
    sites_path = "data/maps/REV_Community_matrix.csv"
    df_sites = pd.read_csv(sites_path)

    species_path = "data/processed/traits_pred_log.csv"
    df_species = pd.read_csv(species_path)
    df_species["accepted_bin"] = df_species["accepted_bin"].str.replace(" ", "_")

    clusters_path = "output/consensus/gmm_error1.0_scl/full_data/ward/final_clusters.csv"
    df_clusters = pd.read_csv(clusters_path, header=None).squeeze("columns")

    taxonomic_path = "data/taxonomic_information.csv"
    df_taxonomic = pd.read_csv(taxonomic_path)
    df_taxonomic["accepted_bin"] = df_taxonomic["accepted_bin"].str.replace(" ", "_")

    grid_coords_path = "data/maps/grid_coordinates.csv"
    df_grid_coords = pd.read_csv(grid_coords_path)

    return df_sites, df_species, df_clusters, df_taxonomic, df_grid_coords


def process_data(df_sites, df_species, df_clusters, df_taxonomic):
    """
    Merge cluster and taxonomic info into site data, then calculate grid-level metrics.
    """
    # attach cluster labels to species table
    df_species = df_species.copy()
    df_species["cluster"] = df_clusters.values

    # merge cluster info into community matrix
    df_sites = df_sites.merge(
        df_species[["accepted_bin", "cluster"]],
        on="accepted_bin",
        how="left"
    )

    # merge taxonomic info
    df_sites = df_sites.merge(
        df_taxonomic[["accepted_bin", "order"]],
        on="accepted_bin",
        how="left"
    )

    output_path = "output/spatial_analysis"
    os.makedirs(output_path, exist_ok=True)

    # optional debug export
    df_sites.to_csv(os.path.join(output_path, "sites_cluster.csv"), index=False)

    # number of unique clusters per grid
    grid_clusters = (
        df_sites.groupby("grid_id")["cluster"]
        .nunique()
        .reset_index(name="nclust")
    )

    # number of unique orders per grid
    grid_orders = (
        df_sites.groupby("grid_id")["order"]
        .nunique()
        .reset_index(name="norder")
    )

    # number of unique species per grid
    grid_species = (
        df_sites.groupby("grid_id")["accepted_bin"]
        .nunique()
        .reset_index(name="nspec")
    )

    # Simpson concentration for clusters
    cluster_simpson = (
        df_sites.groupby("grid_id")["cluster"]
        .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
        .reset_index(name="cluster_simpson")
    )

    # Simpson concentration for species
    species_simpson = (
        df_sites.groupby("grid_id")["accepted_bin"]
        .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
        .reset_index(name="species_simpson")
    )

    # combine all grid-level summaries
    grid_summary = (
        grid_species
        .merge(grid_clusters, on="grid_id", how="outer")
        .merge(grid_orders, on="grid_id", how="outer")
        .merge(cluster_simpson, on="grid_id", how="outer")
        .merge(species_simpson, on="grid_id", how="outer")
    )

    return df_sites, grid_summary


def plot_cluster_distribution(df_summary):
    """Plot how many grids have a certain number of clusters."""
    grid_ncluster_count = df_summary["nclust"].value_counts().reset_index()
    grid_ncluster_count.columns = ["nclust", "count"]
    grid_ncluster_count = grid_ncluster_count.sort_values(by="nclust")

    grid_ncluster_count.plot(kind="bar", x="nclust", y="count")
    plt.xlabel("Number of clusters")
    plt.ylabel("Count")
    plt.title("Number of clusters per grid")
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    plt.show()


def save_output(df_output):
    """Save final grid-coordinate table with derived metrics."""
    output_path = "data/processed"
    os.makedirs(output_path, exist_ok=True)

    out_file = os.path.join(output_path, "Functional_group_results.csv")
    df_output.to_csv(out_file, index=False)
    print(f"Saved output to {out_file}")


def main():
    df_sites, df_species, df_clusters, df_taxonomic, df_grid_coords = load_data()
    df_sites, grid_summary = process_data(df_sites, df_species, df_clusters, df_taxonomic)

    # merge computed metrics onto coordinates
    df_output = df_grid_coords.merge(grid_summary, on="grid_id", how="left")

    # optional: reorder columns
    preferred_order = [
        "grid_id", "Latitude", "Longitude",
        "nspec", "nclust", "norder", "cluster_simpson", "species_simpson"
    ]
    existing_cols = [c for c in preferred_order if c in df_output.columns]
    remaining_cols = [c for c in df_output.columns if c not in existing_cols]
    df_output = df_output[existing_cols + remaining_cols]

    # optional plot
    # plot_cluster_distribution(df_output)

    save_output(df_output)


if __name__ == "__main__":
    main()

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# def load_data():
#     """Load necessary datasets from the specified directory."""
#     sites_path = os.path.join("data/maps/REV_Community_matrix.csv")
#     df_sites = pd.read_csv(sites_path)

#     species_path = os.path.join("data/processed/traits_pred_log.csv")
#     df_species = pd.read_csv(species_path)
#     df_species['accepted_bin'] = df_species['accepted_bin'].str.replace(" ", "_")

#     clusters_path = os.path.join(
#         "output/consensus/gmm_error1.0_scl/full_data/ward/final_clusters.csv"
#     )
#     df_clusters = pd.read_csv(clusters_path, header=None).to_numpy()

#     metrics_path = os.path.join(
#         "data/maps/REV_obs_results_200.csv"
#     )
#     df_metrics = pd.read_csv(metrics_path)

#     # data\taxonomic_information.csv
#     df_taxonomic = pd.read_csv("data/taxonomic_information.csv")
#     df_taxonomic['accepted_bin'] = df_taxonomic['accepted_bin'].str.replace(" ", "_")

#     return df_sites, df_species, df_clusters, df_metrics, df_taxonomic


# def process_data(df_sites, df_species, df_clusters, df_taxonomic):
#     """Merge cluster info into site data and count unique clusters and species per grid."""
#     df_species['cluster'] = df_clusters
#     df_sites = df_sites.merge(df_species[['accepted_bin', 'cluster']], on='accepted_bin', how='left')

#     # merge taxonomic info
#     df_sites = df_sites.merge(df_taxonomic[['accepted_bin', 'order']], on='accepted_bin', how='left')

#     output_path = os.path.join(
#         "output/spatial_analysis"
#     )
#     os.makedirs(output_path, exist_ok=True)
#     # save df_sites to check simpsons calculation
#     df_sites.to_csv(os.path.join(output_path,'sites_cluster.csv'))


#     # Count unique clusters per grid
#     grid_clusters = (
#         df_sites.groupby('grid_id')['cluster']
#         .nunique()
#         .reset_index(name='nclust')
#     )

#     # count unique orders per grid
#     grid_clusters['norder'] = (
#         df_sites.groupby('grid_id')['order']
#         .nunique()
#         .reset_index(name='norder')['norder']
#     )
    


#     # Count unique species per grid
#     grid_species = (
#         df_sites.groupby('grid_id')['accepted_bin']
#         .nunique()
#         .reset_index(name='nspec')
#     )

#     # Simpson's index (as simpson measure)
#     grid_species['cluster_simpson'] = (
#         df_sites.groupby('grid_id')['cluster']
#         .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
#         .reset_index(name='simpson')['simpson']
#     )

#     # Inverse Simpson
#     # grid_species['inverse_simpson'] = 1/(grid_species['simpson'])



#     # Simpson's index (as simpson measure)
#     grid_species['species_simpson'] = (
#         df_sites.groupby('grid_id')['accepted_bin']
#         .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
#         .reset_index(name='simpson')['simpson']
#     )

#     # average species per cluster
#     # grid_species['redun'] = (
#     #             df_sites.groupby('grid_id')['cluster']
#     #     .apply(lambda x: (x.value_counts(normalize=True) ** 2).sum())
#     #     .reset_index(name='simpson')['simpson']

#     # )

#     # Merge the two summaries
#     grid_summary = pd.merge(grid_clusters, grid_species, on='grid_id')
    

#     return df_sites, grid_summary



# def plot_cluster_distribution(df_metrics):
#     """Plot how many grids have a certain number of clusters."""
#     grid_ncluster_count = df_metrics['nclust'].value_counts().reset_index()
#     grid_ncluster_count.columns = ['nclust', 'count']
#     grid_ncluster_count = grid_ncluster_count.sort_values(by='nclust')

#     grid_ncluster_count.plot(kind='bar', x='nclust', y='count')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Count')
#     plt.title('Number of clusters per grid')
#     plt.gcf().set_size_inches(10, 5)
#     plt.tight_layout()
#     plt.show()


# def save_output(df_metrics):
#     """Save the resulting metrics with cluster info to a CSV file."""
#     output_path = os.path.join(
#         "output/spatial_analysis"
#     )
#     os.makedirs(output_path, exist_ok=True)
#     df_metrics.to_csv(os.path.join(output_path, 'REV_obs_results_200_extended.csv'), index=False)
#     print(f"Saved output to {output_path}")


# def main():
#     df_sites, df_species, df_clusters, df_metrics, df_taxonomic = load_data()
#     df_sites, grid_clusters = process_data(df_sites, df_species, df_clusters, df_taxonomic)

#     # Merge cluster count into metrics
#     df_metrics = df_metrics.merge(grid_clusters, on='grid_id', how='left')

#     # Plot distribution
#     # plot_cluster_distribution(df_metrics)

#     # Save final output
#     save_output(df_metrics)


# if __name__ == "__main__":
#     main()
