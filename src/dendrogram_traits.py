import pandas as pd
import numpy as np
import pyarrow as pa
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  # 10pt font size for text elements

from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
import sys

def linewidth():
    return 6.30045

def hierarchical_clustering(distance_matrix, consensus_matrix, num_clusters):
    """ Perform hierarchical clustering and return the clusters and ordered consensus matrix """
    condensed_distance_matrix = squareform(distance_matrix)
    Z = sch.linkage(condensed_distance_matrix, method='ward')
    dendrogram_leaves = sch.leaves_list(Z)
    # plot_dendrogram(Z, num_clusters)
    clusters = sch.fcluster(Z, t=num_clusters, criterion='maxclust')


    # Get the order of the points as they appear in the dendrogram
    dendrogram_leaves = sch.leaves_list(Z)
    
    # Create a new cluster label array that reflects the dendrogram order
    new_cluster_labels = np.zeros_like(clusters)
    for idx, leaf in enumerate(dendrogram_leaves):
        new_cluster_labels[leaf] = clusters[dendrogram_leaves][idx]

    order = np.argsort(clusters)
    ordered_consensus_matrix = consensus_matrix[order, :][:, order]
    return clusters, ordered_consensus_matrix, Z

def plot_dendrogram(Z, num_clusters):
    """ Plot the dendrogram for the hierarchical clustering """
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z, truncate_mode='lastp', p=num_clusters, show_leaf_counts=True)
    plt.title('Dendrogram for Clusters')
    plt.xlabel('Cluster index')
    plt.ylabel('Distance')
    plt.show()

def compute_distance_matrix(consensus_matrix):
    """ Compute the distance matrix from the consensus matrix """
    distance_matrix = 1 - consensus_matrix
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


# def plot_heatmap(Z, num_clusters, centroids, output):
#     """ Plot heatmap for cluster centroids """

#     # Set font size to 14
#     plt.rcParams.update({'font.size': 14})

#     # Plot the Dendrogram and truncate it to show only the last 20 merged clusters
#     fig, axs = plt.subplots(2, figsize=(20, 8), gridspec_kw={'height_ratios': [1, 3]})
#     sch.dendrogram(Z, truncate_mode='lastp', p=num_clusters, ax=axs[0])
#     axs[0].set_axis_off()  # Turn off the axis
#     # sns.heatmap(centroids.T, cmap='RdBu', ax = axs[1], cbar = False)
#     sns.heatmap(centroids.T, cmap='RdBu', center=0, ax=axs[1], cbar=False)
#     plt.xlabel('Functional Groups')
#     plt.ylabel('Traits')

#     plt.subplots_adjust(hspace=0.05)
#     plt.savefig(output)


def plot_heatmap(Z, num_clusters, centroids, output):
    """ Plot heatmap for cluster centroids with capped color bar """
    
    # Cap the values in the centroids at ±3s
    capped_centroids = np.clip(centroids, -3, 3)

    # Create the figure with two axes: one for dendrogram and one for heatmap
    fig, axs = plt.subplots(2, 2, figsize=(linewidth(), linewidth()/2), 
                            gridspec_kw={'width_ratios': [20, .5], 'height_ratios': [1, 3]})
    
    # Plot the dendrogram
    with plt.rc_context({'lines.linewidth': 0.5}):
        sch.dendrogram(Z, truncate_mode='lastp', p=num_clusters, ax=axs[0, 0], 
                                    color_threshold=0, above_threshold_color='k')


    axs[0, 0].set_axis_off()  # Turn off the axis for the dendrogram
    axs[0, 1].set_axis_off()  # Turn off the axis for the empty subplot
    # plt.setp(dendrogram['icoord'], color='black')  # Set this to your desired color
    # plt.setp(dendrogram['dcoord'], color='black')  # Set this to your desired color

    # Plot the heatmap with the color bar
    heatmap = sns.heatmap(capped_centroids.T, cmap='RdBu', center=0, ax=axs[1, 0], cbar_ax=axs[1, 1],
                        xticklabels=True, yticklabels=True)

    # Set labels for the heatmap
    axs[1, 0].set_xlabel('Functional Groups')
    axs[1, 0].set_ylabel('Traits')
    # Customize the tick sizes
    axs[1, 0].tick_params(axis='x', labelsize=6)
    axs[1, 0].tick_params(axis='y', labelsize=6)

    # Customize the color bar with ticks: capped at ±3s
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([-2.99, -2, -1, 0, 1, 2])
    cbar.set_ticklabels([r'$< \mu$ - 3$\sigma$', r'$\mu$ - 2$\sigma$', r'$\mu$ - 1$\sigma$', 
                         r'$\mu$', r'$\mu$ + 1$\sigma$', r'$\mu$ + 2$\sigma$'])
    cbar.ax.tick_params(labelsize=6)

    # Adjust spacing between plots
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_heatmap_with_std(Z, num_clusters, centroids, output):
    """Plot heatmap for cluster centroids with capped color bar and standard deviation plot."""
    
    # Cap the values in the centroids at ±3s
    capped_centroids = np.clip(centroids, -3, 3)
    
    # Calculate the standard deviation for each trait across centroids
    trait_std = np.std(centroids, axis=0)

    # Create the figure with three axes: dendrogram, heatmap, and standard deviation plot
    fig, axs = plt.subplots(2, 3, figsize=(linewidth(), linewidth()/2), 
                            gridspec_kw={'width_ratios': [20, .5, 5], 'height_ratios': [1, 3]})
    
    # Plot the dendrogram
    with plt.rc_context({'lines.linewidth': 0.5}):
        sch.dendrogram(Z, truncate_mode='lastp', p=num_clusters, ax=axs[0, 0], 
                                    color_threshold=0, above_threshold_color='k')
    
    axs[0, 0].set_axis_off()  # Turn off the axis for the dendrogram
    axs[0, 1].set_axis_off()  # Turn off the axis for the empty subplot
    axs[0, 2].set_axis_off()  # Turn off the axis for the standard deviation subplot

    # Plot the heatmap with the color bar
    heatmap = sns.heatmap(capped_centroids.T, cmap='RdBu', center=0, ax=axs[1, 0], cbar_ax=axs[1, 1],
                          xticklabels=True, yticklabels=True)

    # Set labels for the heatmap
    axs[1, 0].set_xlabel('Functional Groups')
    axs[1, 0].set_ylabel('Traits')
    
    # Customize the tick sizes
    axs[1, 0].tick_params(axis='x', labelsize=6)
    axs[1, 0].tick_params(axis='y', labelsize=6)

    # Customize the color bar with ticks: capped at ±3s
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([-2.99, -2, -1, 0, 1, 2])
    cbar.set_ticklabels([r'$< \mu$ - 3$\sigma$', r'$\mu$ - 2$\sigma$', r'$\mu$ - 1$\sigma$', 
                         r'$\mu$', r'$\mu$ + 1$\sigma$', r'$\mu$ + 2$\sigma$'])
    cbar.ax.tick_params(labelsize=6)
    
    # Plot the standard deviation for each trait across centroids
    axs[1, 2].barh(range(len(trait_std)), trait_std, color='gray', edgecolor='black')
    axs[1, 2].set_yticks(range(len(trait_std)))
    axs[1, 2].set_yticklabels([''] * len(trait_std))  # No labels on the y-axis for the bar plot
    axs[1, 2].invert_yaxis()  # Invert y-axis to match the heatmap
    axs[1, 2].set_xlabel('Std Dev', fontsize=8)
    axs[1, 2].tick_params(axis='x', labelsize=6)

    # Adjust spacing between plots
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_determinant_heatmap(det_covs_df, output):
    """ Plot heatmap for cluster determinants """

    # Set font size to 14
    plt.rcParams.update({'font.size': 14})

    # Convert the determinant values to a DataFrame for plotting
    det_matrix = det_covs_df[['Determinant']].T

    # Plot the heatmap
    plt.figure(figsize=(10, 1))
    sns.heatmap(det_matrix, cmap='viridis')
    plt.xlabel('Clusters')
    plt.yticks([])  # Hide y-axis labels since it's just the determinant

    plt.savefig(output)

def plot_heatmap_ordered_traits(Z, num_clusters, centroids, trait_order, output):
    """ Plot heatmap for cluster centroids with different colormaps for trait groups """
    # Flatten the trait_order list to get the original order of traits
    flat_trait_order = [item for sublist in trait_order for item in sublist]
    
    # Rearrange centroids to match the order of traits in flat_trait_order
    reordered_centroids = centroids[flat_trait_order]
    
    # Calculate the number of subplots needed
    num_groups = len(trait_order)
    
    # Calculate height ratios
    trait_counts = [len(group) for group in trait_order]
    height_ratios = [2.5] + [count for count in trait_counts]
    
    # Create the figure and gridspec
    fig = plt.figure(figsize=(25, 2.5 + sum(height_ratios)))  # Adjust the height based on the number of groups
    gs = fig.add_gridspec(num_groups + 1, 1, height_ratios=height_ratios)
    
    # Plot the dendrogram
    ax_dendrogram = fig.add_subplot(gs[0, 0])
    sch.dendrogram(Z, truncate_mode='lastp', p=num_clusters, ax=ax_dendrogram)
    ax_dendrogram.set_axis_off()
    
    # Create custom color maps
    def create_colormap(color):
        return LinearSegmentedColormap.from_list('custom', ['white', color])
    
    colormaps = [
        create_colormap('blue'),
        create_colormap('green'),
        create_colormap('red'),
        create_colormap('purple'),
        create_colormap('orange'),
        create_colormap('grey'),
        create_colormap('deepskyblue'),
        create_colormap('yellowgreen')
    ]
    
    
    # Plot each trait group with a different colormap
    start_col = 0
    for i, group in enumerate(trait_order):
        end_col = start_col + len(group)
        ax_heatmap = fig.add_subplot(gs[i + 1, 0])
        sns.heatmap(reordered_centroids.iloc[:, start_col:end_col].T, cmap=colormaps[i % len(colormaps)], ax=ax_heatmap, cbar=False)
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize = 14)
        ax_heatmap.tick_params(left=False, bottom=False)  # Remove ticks
        # ax_heatmap.set_ylabel('Traits')
        start_col = end_col
    
    plt.subplots_adjust(wspace=0.05, hspace=0.01)
    plt.savefig(output)
    # plt.show()


def plot_centroids(centroids, trait_clusters):
    # Heatmap with different color gradings
    plt.figure(figsize=(14, 8))

    # Define color maps for each cluster
    color_maps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'BuGn', 'YlGnBu', 'PuRd']

    # Plot each cluster separately with its color map
    for i, traits in enumerate(trait_clusters):
        # plt.subplot(1, len(traits), i + 1)
        plt.subplot()
        sns.heatmap(centroids[traits], cmap=color_maps[i], cbar=False, linewidths=0.5)
        # plt.xticks(rotation=45, ha='right')
        # plt.yticks(rotation=0)
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout()
    plt.show()


def max_min_cluster(group):
    return pd.Series({
        'Mean': group.mean(),
        'Std': group.std(),
        'Max': group.max(),
        'Max Cluster': group.idxmax(),
        'Min': group.min(),
        'Min Cluster': group.idxmin()
    })


trait_clusters = [['Leaf density', 'Wood density'],
                   ['Root depth'],
                   ['Specific leaf area', 'Leaf thickness', 'Leaf N per mass'],
                   ['Leaf K per mass', 'Leaf P per mass'],
                   ['Stem conduit diameter', 'Leaf Vcmax per dry mass', 'Stomatal conductance', 'Leaf area'],
                   ['Crown height', 'Crown diameter', 'Tree height'],
                   ['Seed dry mass'],
                   ['Bark thickness', 'Stem diameter']]

shortened_trait_names = {
    'Leaf density': 'Leaf Dens.',
    'Wood density': 'Wood Dens.',
    'Root depth': 'Root Depth',
    'Specific leaf area': 'Spec. Leaf Area',
    'Leaf thickness': 'Leaf Thick.',
    'Leaf N per mass': 'Leaf N/Mass',
    'Leaf K per mass': 'Leaf K/Mass',
    'Leaf P per mass': 'Leaf P/Mass',
    'Stem conduit diameter': 'Stem Cond. Diam.',
    'Leaf Vcmax per dry mass': 'Leaf Vcmax/Mass',
    'Stomatal conductance': 'Stomatal Cond.',
    'Leaf area': 'Leaf Area',
    'Crown height': 'Crown Height',
    'Crown diameter': 'Crown Diam.',
    'Tree height': 'Tree Height',
    'Seed dry mass': 'Seed Mass',
    'Bark thickness': 'Bark Thick.',
    'Stem diameter': 'Stem Diam.'
}

method = sys.argv[1]
consensus_data = 'full_data'
# consensus_data = 'Wood density_Leaf area'
output_dir = os.path.join('output', 'consensus', method, consensus_data)

print('Processing Consensus...')
consensus_matrix = pd.read_parquet(os.path.join(output_dir, 'consensus_matrix.parquet')).values
distance_matrix = 1 - consensus_matrix
np.fill_diagonal(distance_matrix, 0)
num_clusters = 42 # OPTIMAL
df_traits = pd.read_csv('data/traits_pred_log.csv', index_col = 0)
trait_columns = df_traits.columns

clusters, ordered_consensus_matrix, Z = hierarchical_clustering(distance_matrix, consensus_matrix, num_clusters)
# clusters = pd.read_csv(os.path.join(output_dir, 'final_clusters.csv'), header = None).values ## PREASSIGNED
df_traits_standard = df_traits.copy()
df_traits_standard[trait_columns] = StandardScaler().fit_transform(df_traits_standard)

# assign clusters
df_traits['Cluster'] = clusters
df_traits_standard['Cluster'] = clusters
centroids_standard = df_traits_standard.groupby('Cluster').mean()
centroids_standard.columns = [shortened_trait_names.get(trait, trait) for trait in centroids_standard.columns]


# Apply the function to each column in centroids_standard
centroid_stats = centroids_standard.apply(max_min_cluster).T
# # Calculate statistics (mean, std, max, min) for each trait across the clusters in centroids_standard
# centroid_stats = centroids_standard.agg(['mean', 'std', 'max', 'min']).T
# Save the centroid statistics to a CSV file
output_file = 'centroid_trait_statistics.csv'
centroid_stats.to_csv(os.path.join(output_dir,output_file))


# determinants
# Initialize a dictionary to store determinants
plot_det = False
if plot_det:
    det_covs = {}
    # Calculate the determinant of the covariance matrix for each cluster
    for cluster in np.unique(clusters):
        print(f'Determinant {cluster}')
        # Select the data for the current cluster
        cluster_data = df_traits.loc[df_traits['Cluster'] == cluster, trait_columns]
        
        # Compute the covariance matrix
        cov_matrix = np.cov(cluster_data, rowvar=False)
        
        # Calculate the determinant of the covariance matrix
        determinant = np.linalg.det(cov_matrix)
        
        # Store the determinant in the dictionary
        det_covs[cluster] = determinant
    # Convert the dictionary to a DataFrame
    det_covs_df = pd.DataFrame(list(det_covs.items()), columns=['Cluster', 'Determinant'])
    plot_determinant_heatmap(det_covs_df, os.path.join(output_dir, 'images', 'det_covs.pdf'))

plot_heatmap(Z, num_clusters, centroids_standard, os.path.join(output_dir, 'images', 'functional_groups_dendrogram.pdf'))
plot_heatmap_with_std(Z, num_clusters, centroids_standard, os.path.join(output_dir, 'images', 'functional_groups_dendrogram_std.pdf'))
# plot_heatmap_ordered_traits(Z, num_clusters, centroids, trait_clusters, os.path.join(output_dir, 'images', 'functional_groups_dendogram_colored_traits.pdf'))


# # Create a figure with two subplots: one for the dendrogram and one for the heatmap
# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])  # Dendrogram on the left
# ax2 = fig.add_axes([0.3, 0.1, 0.6, 0.8])  # Heatmap on the right

# # Plot the dendrogram
# dendro = sch.dendrogram(Z, orientation='left', ax=ax1, no_labels=True)

# # Reorder the rows in cluster_means to match the dendrogram order
# # dendro_leaves = dendro['leaves']
# # centroids = centroids.iloc[dendro_leaves]

# # Plot the heatmap
# sns.heatmap(centroids, ax=ax2, cmap='viridis', cbar_kws={'orientation': 'horizontal'})

# # Remove the y-axis labels from the heatmap to avoid redundancy
# ax2.set_yticklabels([])
# ax2.set_ylabel('')

# # Show the plot
# plt.show()



