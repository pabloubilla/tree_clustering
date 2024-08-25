import pandas as pd
import numpy as np
import pyarrow as pa
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  

from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from joblib import Parallel, delayed
import sys

def linewidth():
    return 6.30045

def downsample_matrix(matrix, factor):
    """ Downsample a matrix by averaging over factor x factor blocks """
    new_size = matrix.shape[0] // factor, matrix.shape[1] // factor
    downsampled = np.zeros(new_size)
    
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            downsampled[i, j] = np.mean(matrix[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    
    return downsampled

def load_consensus_matrix(output_dir):
    """ Load consensus matrix from a parquet file """
    return pd.read_parquet(os.path.join(output_dir, 'consensus_matrix.parquet')).values

def compute_distance_matrix(consensus_matrix):
    """ Compute the distance matrix from the consensus matrix """
    distance_matrix = 1 - consensus_matrix
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def hierarchical_clustering(distance_matrix, consensus_matrix, num_clusters):
    """ Perform hierarchical clustering and return the clusters and ordered consensus matrix """
    condensed_distance_matrix = squareform(distance_matrix)
    Z = sch.linkage(condensed_distance_matrix, method='ward')
    clusters = sch.fcluster(Z, t=num_clusters, criterion='maxclust')
    order = np.argsort(clusters)
    ordered_consensus_matrix = consensus_matrix[order, :][:, order]
    return clusters, ordered_consensus_matrix

def plot_heatmap(matrix, output_path, title='Heatmap'):
    """ Plot and save heatmap of the matrix """
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(matrix, cmap='Reds', xticklabels=False, yticklabels=False)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Consensus', rotation=270, labelpad=20)
    ax.set_xlabel('Species')
    ax.set_ylabel('Species')
    plt.savefig(output_path)
    plt.clf()

def compute_summary_matrix(consensus_matrix, consensus_labels):
    '''
    Summary of consensus: avg distance between every cluster pair
                          diagonal is avg distance inside cluster
    '''

    num_clusters = len(np.unique(consensus_labels))
    consensus_summary_matrix = np.zeros((num_clusters, num_clusters))
    cluster_sizes = np.zeros(num_clusters)

    for c in range(1, num_clusters+1):
        index_c = np.where(consensus_labels == c)[0]
        cluster_size = len(index_c)
        cluster_sizes[c-1] = cluster_size
        matrix_c = consensus_matrix[index_c, :][:, index_c]
        avg_consensus_c = matrix_c.sum() / (len(index_c)**2 - len(index_c))  # all except diagonal (elements repeat)
        # print(f'Average consensus {c}: {avg_consensus_c}')
        consensus_summary_matrix[c-1, c-1] = avg_consensus_c
        for c_2 in range(c+1, num_clusters+1):
            # compute average consensus between clusters
            index_c_2 = np.where(consensus_labels == c_2)[0]
            matrix_c1_c2 = consensus_matrix[index_c, :][:, index_c_2]
            avg_consensus_c1_c2 = matrix_c1_c2.sum() / (len(index_c) * len(index_c_2))
            consensus_summary_matrix[c-1, c_2-1] = avg_consensus_c1_c2
            consensus_summary_matrix[c_2-1, c-1] = avg_consensus_c1_c2
            # print(f'Average consensus {c}-{c_2} v2: {avg_consensus_c1_c2}')

    return consensus_summary_matrix, cluster_sizes



def plot_summary_heatmap(consensus_summary_matrix, cluster_sizes, output_path,
                            scale_small = 1000, scale_big = 10000, sorted = False):
    
    # Sort clusters by size
    num_clusters = len(cluster_sizes)
    sorted_indices = np.argsort(-cluster_sizes)
    if sorted:
        sorted_matrix = consensus_summary_matrix[sorted_indices, :][:, sorted_indices]
        sorted_cluster_sizes = cluster_sizes[sorted_indices]
    else: 
        sorted_matrix = consensus_summary_matrix.copy()
        sorted_cluster_sizes = cluster_sizes.copy()

    # Normalize cluster sizes to get proportions
    total_size = sum(sorted_cluster_sizes)
    normalized_sizes = sorted_cluster_sizes / total_size

    # Create custom plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Remove plot borders
    ax.axis('off')

    current_x = 0.01
    for i in range(num_clusters):
        current_y = 0.08
        for j in range(num_clusters):
            width = normalized_sizes[i]   # Scaling factor for visual clarity
            height = normalized_sizes[j]  # Scaling factor for visual clarity
            color = plt.cm.Reds(sorted_matrix[i, j])
            rect = plt.Rectangle((current_x, current_y), width, height, facecolor=color, edgecolor='black', linewidth = 0.1)
            ax.add_patch(rect)
            current_y += height
        current_x += width

    # Titles and labels
    ax.set_xlim(0, current_x + 0.01)
    ax.set_ylim(0, current_y + 0.01)
    ax.invert_yaxis()
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.gca().set_aspect('equal', adjustable='box')

    # Add color bar
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average consensus')

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    # Calculate scales
    scale_small_species = scale_small / total_size
    scale_big_species = scale_big / total_size

    # Plot the reference scale line
    ax.plot([0.01, scale_big_species + 0.01], [0.03, 0.03], color='black', linewidth=1.5, linestyle='--')

    # Add legend on top of the line
    ax.text(0.06, 0.015, 'Num. Species', ha='center', va='bottom')

    # Add ticks and labels for the scales
    tick_length = 0.007
    ax.plot([0.01, 0.01], [0.03 - tick_length, 0.03 + tick_length], color='black', linewidth=1.5)
    ax.plot([scale_small_species + 0.01, scale_small_species + 0.01], [0.03 - tick_length, 0.03 + tick_length], color='black', linewidth=1.5)
    ax.text(scale_small_species + 0.01, 0.045, str(scale_small), ha='center', va='top')

    ax.plot([scale_big_species + 0.01, scale_big_species + 0.01], [0.03 - tick_length, 0.03 + tick_length], color='black', linewidth=1.5)
    ax.text(scale_big_species + 0.01, 0.045, str(scale_big), ha='center', va='top')

    plt.tight_layout()
    plt.savefig(output_path)


def plot_summary_heatmap_w_squares(consensus_summary_matrix, cluster_sizes, output_path,
                            specific_sizes = [1000,2000,4000], sorted = False):
    
    # Sort clusters by size
    num_clusters = len(cluster_sizes)
    sorted_indices = np.argsort(-cluster_sizes)
    if sorted:
        sorted_matrix = consensus_summary_matrix[sorted_indices, :][:, sorted_indices]
        sorted_cluster_sizes = cluster_sizes[sorted_indices]
    else: 
        sorted_matrix = consensus_summary_matrix.copy()
        sorted_cluster_sizes = cluster_sizes.copy()

    # Normalize cluster sizes to get proportions
    total_size = sum(sorted_cluster_sizes)
    normalized_sizes = sorted_cluster_sizes / total_size
    # Create custom plot
    fig, ax = plt.subplots(figsize=(.8*linewidth(), .8*linewidth()-1.5))
    current_x = 0

    # for i in range(num_clusters):
    #     current_y = 0
    #     for j in range(num_clusters):
    #         width = normalized_sizes[i]   # Scaling factor for visual clarity
    #         height = normalized_sizes[j] # Scaling factor for visual clarity
    #         color = plt.cm.Reds(sorted_matrix[i, j])
    #         if i == j: 
    #             edgecolor = 'black' 
    #             lw = 0.5
    #         else: 
    #             edgecolor = None
    #             lw = 0
    #         rect = plt.Rectangle((current_x, current_y), width, height, facecolor=color, edgecolor=edgecolor, linewidth=lw)
    #         ax.add_patch(rect)
    #         current_y += height
    #     current_x += width


    # Initialize current_x for drawing
    current_x = 0

    # First, draw the non-diagonal rectangles
    for i in range(num_clusters):
        current_y = 0
        width = normalized_sizes[i]   # Scaling factor for visual clarity
        for j in range(num_clusters):
            height = normalized_sizes[j]  # Scaling factor for visual clarity
            if i != j:
                color = plt.cm.Reds(sorted_matrix[i, j])
                edgecolor = 'whitesmoke'
                lw = 0.3
                rect = plt.Rectangle((current_x, current_y), width, height, facecolor=color, edgecolor=edgecolor, linewidth=lw)
                ax.add_patch(rect)
            current_y += height  # Move to the next y position
        current_x += width # Move to the next x position

    # Reset current_x for diagonal drawing
    current_x = 0
    current_y = 0

    # Now, draw the diagonal rectangles
    for i in range(num_clusters):
        width = normalized_sizes[i]            # Scaling factor for visual clarity
        height = normalized_sizes[i]           # Scaling factor for visual clarity
        color = plt.cm.Reds(sorted_matrix[i, i])
        edgecolor = 'black'
        lw = 0.5
        rect = plt.Rectangle((current_x, current_y), width, height, facecolor=color, edgecolor=edgecolor, linewidth=lw)
        ax.add_patch(rect)
        current_x += width  # Move to the next x position for the next diagonal element
        current_y += height

    # Remove plot borders
    ax.axis('off')
    # Titles and labels
    ax.set_xlim(-0.25, current_x + 0.01)
    ax.set_ylim(0, current_y + 0.3)
    ax.invert_yaxis()
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.gca().set_aspect('equal', adjustable='box')

    # Add color bar
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Average consensus', rotation=270, labelpad=20)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    #### LEGEND SQUARES (Left) ###
    # Define the specific sizes you want for the legend

    # Calculate the corresponding normalized sizes
    normalized_specific_sizes = [size / total_size for size in specific_sizes]

    # Adjusted positions for the legend squares
    x_start = -0.2  # Move further to the left
    starting_y = 1  # Start closer to the top

    # Draw each specified square and annotate
    for i, size in enumerate(normalized_specific_sizes):
        rect = plt.Rectangle((x_start+0.08, starting_y - size), size, size, facecolor='white', edgecolor='black',  linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x_start - 0.15, starting_y - size + size/2 , f'N = {specific_sizes[i]}', 
                ha='left', fontsize = 8)
        starting_y -= size + 0.2  # Increment y position with a bit more spacing

    plt.tight_layout(pad = 0.1)
    plt.savefig(output_path)



def calculate_silhouette_score(distance_matrix, n_clust):
    """ Calculate the silhouette score for a given number of clusters """
    print(f'Calculating for {n_clust} clusters')
    condensed_distance_matrix = squareform(distance_matrix)
    Z = sch.linkage(condensed_distance_matrix, method='ward')
    clusters = sch.fcluster(Z, t=n_clust, criterion='maxclust')
    score = silhouette_score(distance_matrix, clusters, metric='precomputed')
    print(score)
    return score

def analyze_clusters(distance_matrix, consensus_matrix, n_cluster_list, output, method, n_jobs=1):
    """ Analyze different cluster sizes and plot silhouette scores """
    
    output_images = os.path.join(output, 'images')

    if n_jobs == 1:
        # Sequential computation of silhouette scores
        silhouette_score_list = [calculate_silhouette_score(distance_matrix, n_clust) for n_clust in n_cluster_list]
    else:
        # Parallel computation of silhouette scores
        silhouette_score_list = Parallel(n_jobs=n_jobs)(delayed(calculate_silhouette_score)(distance_matrix, n_clust) for n_clust in n_cluster_list)


    # Store results in a DataFrame
    df_silhouette = pd.DataFrame({
        'K': n_cluster_list,
        'Silhouette Score': silhouette_score_list
    })
    
    # Save the DataFrame to CSV
    df_silhouette.to_csv(os.path.join(output, f'silhouette_scores_{method}.csv'), index=False)


    best_num_clusters = n_cluster_list[np.argmax(silhouette_score_list)]
    max_silhouette_score = max(silhouette_score_list)
    
    plt.figure(figsize=(12, 6))  # Set figure size
    plt.plot(n_cluster_list, silhouette_score_list, linestyle='-', color='b')
    plt.xlabel('Number of groups ($K$)')
    plt.ylabel('Silhouette Score')
    
    # Highlight the maximum value with a marker and text
    plt.scatter(best_num_clusters, max_silhouette_score, color='red', zorder=5)
    plt.text(n_cluster_list[-1], max(silhouette_score_list), f'Best K = {best_num_clusters}\nMax Score = {max_silhouette_score:.2f}', 
             horizontalalignment='right', verticalalignment='top', color='red', fontsize=10, 
             bbox=dict(facecolor='lightpink', alpha=0.8, edgecolor='red'))

    plt.savefig(os.path.join(output_images, f'silhouette_score_{method}.pdf'))
    plt.clf()
    
    return best_num_clusters

def save_or_load_summary_matrix(consensus_matrix, clusters, output_dir):
    summary_matrix_file = os.path.join(output_dir, 'summary_consensus_matrix.csv')
    cluster_sizes_file = os.path.join(output_dir, 'cluster_sizes.csv')
    
    # Check if files exist
    if os.path.exists(summary_matrix_file) and os.path.exists(cluster_sizes_file):
        print('Files found. Loading saved data.')
        summary_consensus_matrix = pd.read_csv(summary_matrix_file, index_col=0).values
        cluster_sizes = pd.read_csv(cluster_sizes_file, index_col=0).values.flatten()
    else:
        print('Files not found. Computing data.')
        summary_consensus_matrix, cluster_sizes = compute_summary_matrix(consensus_matrix, clusters)
        pd.DataFrame(summary_consensus_matrix).to_csv(summary_matrix_file)
        pd.DataFrame(cluster_sizes).to_csv(cluster_sizes_file)
    
    return summary_consensus_matrix, cluster_sizes

def main():
    method = sys.argv[1]
    if len(sys.argv) == 3:
        n_jobs = int(sys.argv[2])
    else:
        n_jobs = 1
    
    consensus_data = 'full_data'
    # consensus_data = 'Wood density_Leaf area'
    output_dir = os.path.join('output', 'consensus', method, consensus_data)
    images_dir = os.path.join(output_dir, 'images')

    # clusters to try
    # n_cluster_list = [i*5 for i in range(1,21)]
    n_cluster_list = [i for i in range(2,120)]
    # n_cluster_list = [i for i in [100,200,300,400,500]]
    # n_cluster_list = [2,3,4,5,6]
    
    # Create the images directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    consensus_matrix = load_consensus_matrix(output_dir)
    print('Matrix subsample:')
    print(consensus_matrix[100:110, 100:110])
    # max_values = np.max(np.where(np.eye(consensus_matrix.shape[0], dtype=bool), -np.inf, consensus_matrix), axis=1)
    # sns.displot(max_values)

    distance_matrix = compute_distance_matrix(consensus_matrix)

    print('Analyzing cluster sizes...')
    # Get best nuber of clusters using silhouette score
    best_num_clusters = analyze_clusters(distance_matrix, consensus_matrix, n_cluster_list, output_dir, method, n_jobs)
    print('USING PREDEFINED NUMBER OF CLUSTERS!!!')
    # best_num_clusters = 42
    # best_num_clusters = 50

    print(f'Best number of clusters: {best_num_clusters}')
    print('Performing hierarchical clustering...')
    clusters, ordered_consensus_matrix = hierarchical_clustering(distance_matrix, consensus_matrix, best_num_clusters)

    # save clusters
    pd.DataFrame(clusters).to_csv(os.path.join(output_dir, 'final_clusters.csv'), index=False, header=False)

    # ds_ordered_consensus_matrix = downsample_matrix(ordered_consensus_matrix, 50)
    # print('Shape of downsize: ', ds_ordered_consensus_matrix.shape)

    # print('Plotting heatmap ORDERED...')
    # plot_heatmap(ds_ordered_consensus_matrix, os.path.join(images_dir, f'heatmap_ordered_G{best_num_clusters}.pdf'), 'Heatmap of Ordered Distance Matrix')
    
    # ds_consensus_matrix = downsample_matrix(consensus_matrix, 50)
    # print('Plotting heatmap ORIGINAL...')
    # plot_heatmap(ds_consensus_matrix, os.path.join(images_dir, f'heatmap_G{best_num_clusters}.pdf'), 'Heatmap of Ordered Distance Matrix')

    print('Plotting Summary')
    summary_consensus_matrix, cluster_sizes = save_or_load_summary_matrix(consensus_matrix, clusters, output_dir)
    print(summary_consensus_matrix)
   
    sorted = False
    plot_summary_heatmap(summary_consensus_matrix, cluster_sizes, os.path.join(images_dir, f'heatmap_G{best_num_clusters}_summary{sorted*'sorted'}.pdf'), sorted = False)
    plot_summary_heatmap_w_squares(summary_consensus_matrix, cluster_sizes, os.path.join(images_dir, f'heatmap_G{best_num_clusters}_squares.pdf'), sorted = False)

    # Save cluster information to a dataframe
    cluster_info = {
        'Cluster': range(1, best_num_clusters + 1),
        'Size': cluster_sizes,
        'Avg Consensus': np.diag(summary_consensus_matrix)
    }
    cluster_df = pd.DataFrame(cluster_info)
    cluster_df.to_csv(os.path.join(output_dir, 'clusters_info.csv'), index=False)

if __name__ == '__main__':
    main()



# import pandas as pd
# import numpy as np
# import pyarrow as pa
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score

# import scipy.cluster.hierarchy as sch
# from scipy.spatial.distance import squareform


# def downsample_matrix(matrix, factor):
#     """ Downsample a matrix by averaging over factor x factor blocks """
#     # Compute the size of the downsampled matrix
#     new_size = matrix.shape[0] // factor, matrix.shape[1] // factor
#     downsampled = np.zeros(new_size)
    
#     for i in range(new_size[0]):
#         for j in range(new_size[1]):
#             downsampled[i, j] = np.mean(matrix[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    
#     return downsampled


# # output directory
# method = 'hdbscan_error0.5'
# output_dir = os.path.join('output',  'consensus', method, 'full_data')
# output_images = os.path.join('output', 'images')
# # read consensus matrix
# consensus_matrix = pd.read_parquet(os.path.join(output_dir, 'consensus_matrix.parquet')).values
# # print a portion of the matrix
# print('Matrix subsample:')
# print(consensus_matrix[100:110,100:110]) ## something is going on with the diagonal (it should be 1 or 0)
# distance_matrix = 1 - consensus_matrix
# np.fill_diagonal(distance_matrix, 0)

# # ### plot distnce matrix
# # sns.heatmap(downsized_consensus, cmap='Reds')
# # plt.savefig(os.path.join(output_images,'consensus_matrix_heatmap.pdf'))

# # Convert to condensed distance matrix for clustering
# condensed_distance_matrix = squareform(distance_matrix)


# # n_cluster_list = [5,10,15,20,25,30,35,40]

# # silhouette_score_list = [] 
# # for n_clust in n_cluster_list:
# #     print(f'Calculating for {n_clust} clusters')
# #     Z = sch.linkage(condensed_distance_matrix, method = 'ward')
# #     clusters = sch.fcluster(Z, t = n_clust, criterion='maxclust')
# #     score = silhouette_score(distance_matrix, clusters, metric='precomputed')
# #     print(score)
# #     silhouette_score_list.append(score)

# # plt.plot(n_cluster_list, silhouette_score_list)
# # plt.savefig(os.path.join(output_images, f'silhoutte_score_{method}.pdf'))
# # plt.clf()




# num_clusters = 20

# # Perform hierarchical clustering
# print('Performing hierarchical...')
# Z = sch.linkage(condensed_distance_matrix, method='ward')

# # Retrieve cluster labels at a given cutoff (not super sure how this works)
# clusters = sch.fcluster(Z, t = num_clusters, criterion='maxclust')

# # Order the distance matrix by clusters
# order = np.argsort(clusters)  # This gives indices that sort the clusters
# ordered_consensus_matrix = consensus_matrix[order, :][:, order]

# ds_ordered_consensus_matrix = downsample_matrix(ordered_consensus_matrix, 50)
# print('Shape of downsize: ', ds_ordered_consensus_matrix.shape)
# # Plot the heatmap
# print('Plotting...')
# plt.figure(figsize=(8, 5))
# sns.heatmap(ds_ordered_consensus_matrix, cmap='Reds', xticklabels=False, yticklabels=False)
# plt.title('Heatmap of Ordered Distance Matrix')
# plt.savefig(os.path.join(output_images, f'heatmap_ordered_{method}.pdf'))

