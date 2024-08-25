import pandas as pd
import pyarrow as pa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from scipy.stats import pearsonr
from scipy.stats import ks_2samp

import os
# from Bio import Phylo
from analyse_consensus import compute_distance_matrix, hierarchical_clustering
from multiprocessing import Pool

import sys

def linewidth():
    return 6.30045


def plot_phylogenetic_distance(n_species_genus, mean_phylo_dist_genus, 
                               n_species_genus_cluster, mean_phylo_dist_genus_cluster,
                               n_species_family, mean_phylo_dist_family, 
                               n_species_family_cluster, mean_phylo_dist_family_cluster,
                               n_species_order, mean_phylo_dist_order, 
                               n_species_order_cluster, mean_phylo_dist_order_cluster,
                               output_dir):
    fig, axs = plt.subplots(1, 9, figsize=(24, 5), sharey=True)

    # Function to hide axis
    def hide_axis(ax):
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Set the visibility of the extra subplots to False
    for i in [2, 5, 8]:
        hide_axis(axs[i])

    ## GENUS ##
    axs[0].scatter(np.log10(n_species_genus), mean_phylo_dist_genus, label='Genus',
                   color='blue', alpha=0.5, edgecolors='none')
    axs[1].scatter(np.log10(n_species_genus_cluster), mean_phylo_dist_genus_cluster, label='Consensus Clustering',
                   color='blue', alpha=0.5, edgecolors='none')

    # Line with average
    mean_phylo_dist_genus_cluster_avg = np.mean(mean_phylo_dist_genus_cluster)
    axs[1].axhline(mean_phylo_dist_genus_cluster_avg, color='black', linestyle='--')
    mean_phylo_dist_genus_avg = np.mean(mean_phylo_dist_genus)
    axs[0].axhline(mean_phylo_dist_genus_avg, color='black', linestyle='--')

    # Titles and labels
    axs[0].set_title('Genus')
    axs[1].set_title('Consensus Clustering Genus')
    axs[0].set_xlabel('Log10(Number of Species)')
    axs[1].set_xlabel('Log10(Number of Species)')
    axs[0].set_ylabel('Mean Phylogenetic Distance')

    ## FAMILY ##
    axs[3].scatter(np.log10(n_species_family), mean_phylo_dist_family, label='Family',
                   color='red', alpha=0.5, edgecolors='none')
    axs[4].scatter(np.log10(n_species_family_cluster), mean_phylo_dist_family_cluster, label='Consensus Clustering',
                   color='red', alpha=0.5, edgecolors='none')

    # Line with average
    mean_phylo_dist_family_cluster_avg = np.mean(mean_phylo_dist_family_cluster)
    axs[4].axhline(mean_phylo_dist_family_cluster_avg, color='black', linestyle='--')
    mean_phylo_dist_family_avg = np.mean(mean_phylo_dist_family)
    axs[3].axhline(mean_phylo_dist_family_avg, color='black', linestyle='--')

    # Titles and labels
    axs[3].set_title('Family')
    axs[4].set_title('Consensus Clustering Family')
    axs[3].set_xlabel('Log10(Number of Species)')
    axs[4].set_xlabel('Log10(Number of Species)')

    ## ORDER ##
    axs[6].scatter(np.log10(n_species_order), mean_phylo_dist_order, label='Order',
                   color='green', alpha=0.5, edgecolors='none')
    axs[7].scatter(np.log10(n_species_order_cluster), mean_phylo_dist_order_cluster, label='Consensus Clustering',
                   color='green', alpha=0.5, edgecolors='none')

    # Line with average
    mean_phylo_dist_order_cluster_avg = np.mean(mean_phylo_dist_order_cluster)
    axs[7].axhline(mean_phylo_dist_order_cluster_avg, color='black', linestyle='--')
    mean_phylo_dist_order_avg = np.mean(mean_phylo_dist_order)
    axs[6].axhline(mean_phylo_dist_order_avg, color='black', linestyle='--')

    # Titles and labels
    axs[6].set_title('Order')
    axs[7].set_title('Consensus Clustering Order')
    axs[6].set_xlabel('Log10(Number of Species)')
    axs[7].set_xlabel('Log10(Number of Species)')

    # Adjust the layout to add space between pairs of plots
    fig.subplots_adjust(wspace=0.1)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save figure
    plt.savefig(os.path.join(output_dir, 'phylo_vs_trait.pdf'))

def compute_group_dist(df_tax, tax_name, dist_matrix):
    unique_tax = df_tax[tax_name].unique()
    n_tax = len(unique_tax)
    tax_n_species_list = []
    mean_dist_list = []
    cv_dist_list = []

    for g in unique_tax:
        # get species in genus
        species = df_tax[df_tax[tax_name] == g]['accepted_bin'].tolist()
        n_species_g = len(species)
        if n_species_g == 0 or n_species_g == 1:
            continue
        # replace space with _
        species_ = [s.replace(' ', '_') for s in species]
        # get phylogenetic distance between species
        phylo_dist_g = dist_matrix.loc[species_, species_]
        # get the upper triangle of the phylogenetic distance matrix, excluding the diagonal
        phylo_dist_upper = phylo_dist_g.where(np.triu(np.ones(phylo_dist_g.shape), k=1).astype(bool))
        # flatten the upper triangle matrix and remove NaN values
        upper_values = phylo_dist_upper.values[np.triu_indices(n_species_g, k=1)]
        # get the mean phylogenetic distance
        mean_k = np.mean(upper_values)
        # get the standard deviation of phylogenetic distances
        std_k = np.std(upper_values)
        # calculate the coefficient of variation
        cv_k = std_k / mean_k if mean_k != 0 else 0

        tax_n_species_list.append(n_species_g)
        mean_dist_list.append(mean_k)
        cv_dist_list.append(cv_k)
    
    return n_tax, tax_n_species_list, mean_dist_list, cv_dist_list

def create_order_cluster_heatmap(tax, output):
    # Calculate the counts of each combination 
    count_matrix = pd.crosstab(tax['order'], tax['cluster'])

    # Calculate the proportions
    proportion_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

    width, height = proportion_matrix.shape

    # Calculate aspect ratio to keep the cells square
    aspect_ratio = width / height

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(linewidth(), linewidth() / aspect_ratio))

    # Create a divider for the existing axes
    divider = make_axes_locatable(ax)

    # Append axes for the colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Plot the heatmap
    sns.heatmap(proportion_matrix, 
                cmap=sns.light_palette('lightseagreen', as_cmap=True), 
                cbar=True, 
                cbar_ax=cax,  # Place the colorbar in the new axes
                linewidths=.5, 
                linecolor='black',
                xticklabels=True, 
                yticklabels=True,
                ax=ax)

    ax.set_xlabel('Functional Groups')
    ax.set_ylabel('Taxonomic Groups (Order)')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6, rotation=0)

    plt.tight_layout(pad=0.1)
    plt.savefig(output)
    plt.close()  # Close the figure after saving to avoid display issues

def create_order_cluster_heatmap_rev(tax, output):
    # Calculate the counts of each combination 
    count_matrix = pd.crosstab(tax['order'], tax['cluster'])

    # Calculate the proportions normalized by columns
    proportion_matrix = count_matrix.div(count_matrix.sum(axis=0), axis=1)

    width, height = proportion_matrix.T.shape

    # Calculate aspect ratio to keep the cells square
    aspect_ratio = width / height

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(linewidth(), linewidth()*0.8 / aspect_ratio))

    # Create a divider for the existing axes
    divider = make_axes_locatable(ax)

    # Append axes for the colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Plot the heatmap
    sns.heatmap(proportion_matrix.T, 
                cmap=sns.light_palette('palevioletred', as_cmap=True), 
                cbar=True, 
                cbar_ax=cax,  # Place the colorbar in the new axes
                linewidths=.5, 
                linecolor='black',
                xticklabels=True, 
                yticklabels=True,
                ax=ax)

    ax.set_xlabel('Taxonomic Groups (Order)')
    ax.set_ylabel('Functional Groups')
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6, rotation=0)

    plt.tight_layout(pad=0.1)
    plt.savefig(output)
    plt.close()  # Close the figure after saving to avoid display issues

def create_combined_heatmap(tax, output):
    # Calculate the counts of each combination
    count_matrix = pd.crosstab(tax['order'], tax['cluster'])

    # Calculate the proportions
    proportion_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

    # Highlight rows
    highlighted_row_index = np.argwhere((proportion_matrix > 1.2).any(axis=1)).T[0]
    print('highlight rows')
    print(highlighted_row_index)

    # Calculate the proportions normalized by columns
    proportion_matrix_rev = count_matrix.div(count_matrix.sum(axis=0), axis=1)

    # Highlight columns
    highlighted_column_index = np.argwhere((proportion_matrix_rev > 1.2).any(axis=0)).T[0]
    print('highlight columns')
    print(highlighted_column_index)

    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(26, 16), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1})  # Adjusted height for better readability

    # Plot the first heatmap
    sns.heatmap(proportion_matrix, cmap='Blues', cbar=True, linewidths=.5, linecolor='black', ax=axs[0], cbar_kws={'orientation': 'horizontal', 'pad': 0.07})

    for ix in highlighted_row_index:
        axs[0].add_patch(plt.Rectangle((0, ix), proportion_matrix.shape[1], 1,
                            fill=False, edgecolor='gold', linewidth=2))

    axs[0].set_xlabel('Functional Groups', fontsize=14)
    axs[0].set_ylabel('Taxonomic Groups (Orders)', fontsize=14)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=14)  # Rotate x-axis labels for better readability
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0, fontsize=14)  # Rotate y-axis labels to horizontal for better readability

    # Plot the second heatmap
    sns.heatmap(proportion_matrix_rev, cmap='Reds', cbar=True, linewidths=.5, linecolor='black', ax=axs[1], cbar_kws={'orientation': 'horizontal', 'pad': 0.07})

    for jx in highlighted_column_index:
        axs[1].add_patch(plt.Rectangle((jx, 0), 1, proportion_matrix_rev.shape[0],
                            fill=False, edgecolor='gold', linewidth=2))

    axs[1].set_xlabel('Functional Groups', fontsize=14)
    axs[1].set_ylabel('', fontsize=14)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=14)  # Rotate x-axis labels for better readability
    axs[1].set_yticklabels([])  # Remove y-axis labels on the second heatmap

    # # Adjust color bars
    # for ax in axs:
    #     cbar = ax.collections[0].colorbar
    #     cbar.ax.tick_params(labelsize=14)
    #     cbar.ax.set_aspect(30)  # Make the color bar thicker

    # Adjust layout to minimize white space
    # Adjust layout to prevent cutting off of y-tick labels
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])  # Added margin on the left
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0)  # Increased left margin for y-tick labels
    plt.savefig(output)

def plot_phylo_vs_cons(phylo_dist_cluster, consensus_cluster,
                    phylo_dist_order, consensus_dist_order,
                    output):
    # Create the scatter plot
    plt.figure(figsize=(10, 8))


    # Scatter plot for the second pair of lists with pastel colors
    plt.scatter(phylo_dist_order, consensus_dist_order, 
    color='lightseagreen', s=30, alpha = 0.8,  label='Taxonomic Groups (Order)')

    # Scatter plot for the first pair of lists with pastel colors
    plt.scatter(phylo_dist_cluster, consensus_cluster, 
    color='palevioletred', s=30, alpha = 0.8, label='Functional Groups')

    # Add labels and title
    plt.xlabel('Mean Phylogenetic Distance', fontsize=14)
    plt.ylabel('Mean Consensus', fontsize=14)
    plt.legend(fontsize = 14, loc = 'best')

    # Show the plot
    plt.savefig(os.path.join(output, 'phylo_vs_consensus.pdf'))
    # plt.show()



def plot_phylo_vs_cons_arrows(phylo_dist_cluster, consensus_cluster,
                    phylo_dist_order, consensus_dist_order,
                    output):
    # Create the scatter plot
    plt.figure(figsize=(10, 8))

    # Scatter plot for the second pair of lists with pastel colors
    plt.scatter(phylo_dist_order, consensus_dist_order, 
    color='lightseagreen', s=30, alpha=0.8,  label='Taxonomic Groups (Order)')

    # Scatter plot for the first pair of lists with pastel colors
    plt.scatter(phylo_dist_cluster, consensus_cluster, 
    color='palevioletred', s=30, alpha=0.8, label='Functional Groups')

    # Add labels and title
    plt.xlabel('Mean Phylogenetic Distance', fontsize=14)
    plt.ylabel('Mean Consensus', fontsize=14)
    plt.legend(fontsize=14, loc='best')

    # Get axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Calculate positions for annotations
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # Add arrows and text to indicate evolutionary forces
    plt.annotate('Divergent Evolution', xy=(x_max * 0.9, y_max * 0.9), 
                 xytext=(x_max * 0.75, y_max * 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)

    plt.annotate('Parallel Evolution', xy=(x_min * 1.1, y_max * 0.9), 
                 xytext=(x_min * 1.25, y_max * 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)

    plt.annotate('Convergent Evolution', xy=(x_min * 1.1, y_min * 1.1), 
                 xytext=(x_min * 1.25, y_min * 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)

    plt.annotate('Drift/Noise', xy=(x_max * 0.9, y_min * 1.1), 
                 xytext=(x_max * 0.75, y_min * 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)

    # Save the plot as a PDF
    plt.savefig(os.path.join(output, 'phylo_vs_consensus_arrows.pdf'))
    # plt.show()

def plot_phylo_vs_cons_text(phylo_dist_cluster, consensus_cluster,
                    phylo_dist_order, consensus_dist_order,
                    output):
    # Create the scatter plot
    plt.figure(figsize=(10, 8))

    # Scatter plot for the second pair of lists with pastel colors
    plt.scatter(phylo_dist_order, consensus_dist_order, 
    color='lightseagreen', s=30, alpha=0.8,  label='Taxonomic Groups (Order)')

    # Scatter plot for the first pair of lists with pastel colors
    plt.scatter(phylo_dist_cluster, consensus_cluster, 
    color='palevioletred', s=30, alpha=0.8, label='Functional Groups')

    # Add labels and title
    plt.xlabel('Mean Phylogenetic Distance', fontsize=14)
    plt.ylabel('Mean Consensus', fontsize=14)
    plt.legend(fontsize=14, loc='best')

    # Get axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Calculate positions for annotations
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # Add text to indicate evolutionary forces
    plt.text(x_max * 0.8, y_max * 0.9, 'Divergent Evolution', fontsize=14, ha='center', weight='bold')
    plt.text(x_min * 1.2, y_max * 0.9, 'Parallel Evolution', fontsize=14, ha='center', weight='bold')
    plt.text(x_min * 1.2, y_min * 1.1, 'Convergent Evolution', fontsize=14, ha='center', weight='bold')
    plt.text(x_max * 0.8, y_min * 1.1, 'Drift/Noise', fontsize=14, ha='center', weight='bold')

    # Save the plot as a PDF
    plt.savefig(os.path.join(output, 'phylo_vs_consensus_text_only.pdf'))






    # # Create the figure and axes for the plot
    # plt.figure(figsize=(10, 6))

    # # Create the overlapping histogram (or distribution plot) for mean phylogenetic distances
    # # sns.histplot(mean_phylo_dist_order_cluster, color='palevioletred', kde=True, label='Cluster Order', bins=10)
    # # sns.histplot(mean_phylo_dist_order, color='lightseagreen', kde=True, label='Order', bins=10)

    # # Create the KDE plot for mean phylogenetic distances
    # sns.kdeplot(mean_phylo_dist_order_cluster, color='palevioletred', shade=True, label='Cluster Order')
    # sns.kdeplot(mean_phylo_dist_order, color='lightseagreen', shade=True, label='Order')

    # # Add labels and title
    # plt.xlabel('Phylogenetic Distance', fontsize=14)
    # plt.legend()

    # # Show the plot
    # plt.savefig(os.path.join(output, 'hist_phylo.pdf'))


def random_clusters(df, cluster_sizes, column_name):
    # Shuffle the indices of the DataFrame
    shuffled_indices = np.random.permutation([i for i in range(len(df))])

    # Split the shuffled indices according to the cluster sizes
    cluster_labels = np.zeros(len(df), dtype=int)
    start = 0
    for cluster_id, size in enumerate(cluster_sizes):
        end = start + size
        cluster_labels[shuffled_indices[start:end]] = cluster_id
        start = end

    # Add the cluster labels to the DataFrame
    df[column_name] = cluster_labels

#### MAIN PLOTS
# def plot_dist_means(phylo,random,cluster,output):
#     plt.figure(figsize=(linewidth()*0.8, linewidth()*0.4))
#     sns.kdeplot(phylo, color='lightseagreen', shade=True, label='Taxonomic Groups (Order)')
#     sns.kdeplot(cluster, color='palevioletred', shade=True, label='Functional Groups')
#     sns.kdeplot(random, color='orange', shade=True, label='Random Groups')

#     # Add labels and title
#     plt.xlabel('Mean Phylogenetic Distance')
#     plt.xlim(-20, 300)
#     plt.legend(loc = 'upper left')
#     plt.tight_layout(pad = 0.1)
#     plt.savefig(output)



#### KS test to check they are different
def perform_ks_test(phylo, random, cluster):
    # Perform KS test between phylo and random
    ks_phylo_random = ks_2samp(phylo, random)
    print(f"KS test between Taxonomic Groups (Order) and Random Groups:\n"
          f"  KS Statistic: {ks_phylo_random.statistic}\n"
          f"  P-value: {ks_phylo_random.pvalue}\n")

    # Perform KS test between phylo and cluster
    ks_phylo_cluster = ks_2samp(phylo, cluster)
    print(f"KS test between Taxonomic Groups (Order) and Functional Groups:\n"
          f"  KS Statistic: {ks_phylo_cluster.statistic}\n"
          f"  P-value: {ks_phylo_cluster.pvalue}\n")

    # Perform KS test between random and cluster
    ks_random_cluster = ks_2samp(random, cluster)
    print(f"KS test between Random Groups and Functional Groups:\n"
          f"  KS Statistic: {ks_random_cluster.statistic}\n"
          f"  P-value: {ks_random_cluster.pvalue}\n")

def plot_dist_means(phylo, random, cluster, output):

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2.5]},
                                   figsize=(linewidth()*0.8, linewidth()*0.4))

    # Plot the KDEs on the lower axis
    sns.kdeplot(phylo, color='lightseagreen', shade=True, ax=ax2, label='Taxonomic Groups (Order)')
    sns.kdeplot(cluster, color='palevioletred', shade=True, ax=ax2, label='Functional Groups')
    sns.kdeplot(random, color='orange', shade=True, ax=ax2, label='Random Groups')

    # Set x and y limits for the lower plot
    ax2.set_xlim(-20, 300)
    ax2.set_ylim(0, 0.02)  # Set the y-limit for better visualization

    # Format the y-axis ticks to show 2 decimals
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_ticks([0, 0.01, 0.02])

    # Plot the KDEs on the upper axis with only the Random Group
    sns.kdeplot(random, color='orange', shade=True, ax=ax1)
    
    # Set y-limit for the upper plot and remove the upper plot's x-axis
    ax1.set_ylim(0.02, 0.16)
    # ax1.spines['bottom'].set_visible(False)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax1.yaxis.set_ticks([0.08,0.16])
    ax1.xaxis.set_visible(False)
    # ax1.yaxis.set_visible(False)  # Hide y-axis labels for the top plot
    ax1.yaxis.set_tick_params(labelleft=True)  # Show the y-ticks
    ax1.yaxis.set_label_text('')  # Remove the y-label text

    # Add diagonal cut lines between the two y-axes
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the lower axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # bottom lines
    ax2.spines['top'].set_linestyle((0, (5, 10)))
    ax1.spines['bottom'].set_linestyle((0, (5, 10)))

    # Labels and title
    ax2.set_xlabel('Mean Phylogenetic Distance')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper left')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(output)
    

    perform_ks_test(phylo, random, cluster)

def plot_phylo_vs_cons_w_random(phylo_dist_cluster, consensus_cluster,
                    phylo_dist_order, consensus_dist_order,
                    phylo_random, consensus_random,
                    output):
    # Create the scatter plot
    plt.figure(figsize=(linewidth()*0.8, linewidth()*0.6))


    # Scatter plot for the second pair of lists with pastel colors
    plt.scatter(phylo_dist_order, consensus_dist_order, 
    color='lightseagreen', s=10, alpha = 0.8,  label='Taxonomic Groups (Order)', marker = 's')

    # Scatter plot for the first pair of lists with pastel colors
    plt.scatter(phylo_dist_cluster, consensus_cluster, 
    color='palevioletred', s=10, alpha = 0.8, label='Functional Groups', marker = '^')

    # Scatter plot for randoms
    plt.scatter(phylo_random, consensus_random, 
    color='orange', s=10, alpha = 0.3, label='Random Groups', marker = 'o')

    # Add labels and title
    plt.xlabel('Mean Phylogenetic Distance')
    plt.ylabel('Mean Consensus')
    # plt.legend(loc = 'best')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    # Add white space around the axes by extending the limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.xlim(-20, 300)
    plt.ylim(y_min - (y_max - y_min) * 0.15, y_max + (y_max - y_min) * 0.05)
    print(plt.xlim(), 'THIS IS THE LIMIT')



    # Add text to indicate evolutionary forces
    plt.text(x_max - 20, y_max - 0.05, '(Divergent Evolution)', fontsize=6, ha='right', weight='bold')
    plt.text(x_min +20, y_max - 0.05, '(Parallel Evolution)', fontsize=6, ha='left', weight='bold')
    plt.text(x_min +20, y_min  - 0.05, '(Convergent Evolution)', fontsize=6, ha='left', weight='bold')
    plt.text(x_max - 20, y_min  - 0.05, '(Drift/Noise)', fontsize=6, ha='right', weight='bold')

    # Set the ticks to only show positive values
    plt.xticks([tick for tick in plt.xticks()[0] if tick >= 0])
    plt.yticks([tick for tick in plt.yticks()[0] if tick >= 0])

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout(pad = 0.1)

    # Show the plot
    plt.savefig(os.path.join(output, 'phylo_vs_consensus_w_random.pdf'))
    # plt.show()






# def process_tax_group(args):
#     df_tax, tax_name, g, phylo_dist = args
#     species = df_tax[df_tax[tax_name] == g]['accepted_bin'].tolist()
#     n_species_g = len(species)
#     if n_species_g == 0 or n_species_g == 1:
#         return None
#     species_ = [s.replace(' ', '_') for s in species]
#     phylo_dist_g = phylo_dist.loc[species_, species_]
#     mean_k = np.sum(phylo_dist_g.values) / (n_species_g * (n_species_g - 1))
#     return n_species_g, mean_k

# def dist_tax(df_tax, tax_name, phylo_dist, num_threads=1):
#     unique_tax = df_tax[tax_name].unique()
#     n_tax = len(unique_tax)

#     args_list = [(df_tax, tax_name, g, phylo_dist) for g in unique_tax]
    
#     tax_n_species_list = []
#     tax_mean_phylo_dist_list = []

#     if num_threads == 1:
#         # Process in series
#         for args in args_list:
#             result = process_tax_group(args)
#             if result:
#                 n_species_g, mean_k = result
#                 tax_n_species_list.append(n_species_g)
#                 tax_mean_phylo_dist_list.append(mean_k)
#     else:
#         # Process in parallel
#         with Pool(num_threads) as pool:
#             results = pool.map(process_tax_group, args_list)

#         for result in results:
#             if result:
#                 n_species_g, mean_k = result
#                 tax_n_species_list.append(n_species_g)
#                 tax_mean_phylo_dist_list.append(mean_k)
    
#     return n_tax, tax_n_species_list, tax_mean_phylo_dist_list
if __name__ == '__main__':

    method = sys.argv[1]
    # method = 'gmm_error1.0_rnd'
    consensus_data = 'full_data'
    output_dir = os.path.join('output', 'consensus', method, consensus_data)
    N_random = 20
    print(f'Calculating {N_random} for random aggregations')
    if N_random == 1:
        print('REMEMBER TO RUN AGAIN WITH MORE RANDOM')

    output_phylo = os.path.join('output', 'consensus', method, consensus_data, 'images', 'phylo')
    os.makedirs(output_phylo, exist_ok = True)

    # read C:\Users\pablo\OneDrive\Desktop\tree_clustering\data\traits_pred_log.csv
    df_traits = pd.read_csv('data/traits_pred_log.csv', index_col = 0)
    species_list = list(df_traits.index)
    species_list_ = [s.replace(' ', '_') for s in species_list]
    # read taxonomic_information.csv
    tax = pd.read_csv('data/taxonomic_information.csv')
    tax = tax[tax['accepted_bin'].isin(species_list)]

    # C:\Users\pablo\OneDrive\Desktop\tree_clustering\output\imputed_data\gaussian_mixture\clusters_200.csv
    # clusters = pd.read_csv('output/complete_data/gaussian_mixture_full/clusters_1000.csv',
    #                        index_col=0)
    clusters = pd.read_csv(f'output/consensus/{method}/full_data/final_clusters.csv',
                            names = ['cluster'])
    clusters['accepted_bin'] = species_list

    tax = tax.merge(clusters, on='accepted_bin', how='left') # add clusters to tax
  
    cluster_sizes = clusters['cluster'].value_counts().to_list()
    # # read tree newick file
    # tree_path = ('data/phy_tree_BGCI_full.newick')
    # tree = Phylo.read(tree_path, 'newick')

    # read C:\Users\pablo\OneDrive\Desktop\tree_clustering\data\phylogenetic_distance_matrix_BGCI_full.arrow
    phylo_dist = pd.read_feather('data/phylogenetic_distance_matrix_BGCI_full.arrow')
    phylo_dist.index = phylo_dist.columns
    phylo_dist = phylo_dist.loc[species_list_, species_list_]


    # read consensus matrix
    consensus_matrix = pd.read_parquet(os.path.join(output_dir, 'consensus_matrix.parquet')).values
    distance_matrix = compute_distance_matrix(consensus_matrix)
    df_consensus = pd.DataFrame(consensus_matrix, index = species_list_, columns = species_list_)


    ### THIS IS FOR CORRELATION, Not used now
    # ### take out diagonal for correlation
    # mask = np.eye(consensus_matrix.shape[0], dtype=bool)
    # flat_consensus = consensus_matrix[~mask]
    # flat_phylo_dist = phylo_dist.values[~mask]

    # # Compute correlation
    # # Calculate the correlation coefficient and p-value
    # corr_cons_phylo, p_value = pearsonr(flat_consensus, flat_phylo_dist)
    # # Print the results
    # print(f"Correlation coefficient: {corr_cons_phylo}")
    # print(f"P-value: {p_value}")

    # # genus
    # print('Calculating distances for genus')
    # n_genus, n_species_genus, mean_phylo_dist_genus = dist_tax(tax, 'genus', phylo_dist)
    # print(n_genus)
    # # family
    # print('Calculating distances for family')
    # n_family, n_species_family, mean_phylo_dist_family = dist_tax(tax, 'family', phylo_dist)
    # print(n_family)
    # # order
    print('Calculating distances for order')
    n_order, n_species_order, mean_phylo_order, cv_phylo_dist_order = compute_group_dist(tax, 'order', phylo_dist)
    _, _, mean_consensus_order, _ = compute_group_dist(tax, 'order', df_consensus)
    print(n_order)

    # # genus
    # clusters, _ = hierarchical_clustering(distance_matrix, consensus_matrix, n_genus)
    # tax['cluster_genus'] = clusters
    # print(f'Calculating distances for trait clusters {n_genus}')
    # _, n_species_genus_cluster, mean_phylo_dist_genus_cluster = dist_tax(tax, 'cluster_genus', phylo_dist)
    # # family
    # clusters, _ = hierarchical_clustering(distance_matrix, consensus_matrix, n_family)
    # tax['cluster_family'] = clusters  
    # print(f'Calculating distances for trait clusters {n_family}')
    # _, n_species_family_cluster, mean_phylo_dist_family_cluster = dist_tax(tax, 'cluster_family', phylo_dist)
    # order
    # clusters, _ = hierarchical_clustering(distance_matrix, consensus_matrix, n_order)
    # tax['cluster_order'] = clusters['cluster']
    print(f'Calculating distances for trait clusters {n_order}')
    _, _, mean_phylo_order_cluster, cv_phylo_order_cluster = compute_group_dist(tax, 'cluster', phylo_dist)
    _, _, mean_consensus_order_cluster, cv_consensus_order_cluster = compute_group_dist(tax, 'cluster', df_consensus)

    #### Plot phylo and consensus ###
    # plot_phylo_vs_cons(mean_phylo_order_cluster, mean_consensus_order_cluster,
    #                 mean_phylo_order, mean_consensus_order,
    #                 os.path.join(output_dir, 'images'))
    # plot_phylo_vs_cons_arrows(mean_phylo_order_cluster, mean_consensus_order_cluster,
    #                 mean_phylo_order, mean_consensus_order,
    #                 os.path.join(output_dir, 'images'))
    # plot_phylo_vs_cons_text(mean_phylo_order_cluster, mean_consensus_order_cluster,
    #                 mean_phylo_order, mean_consensus_order,
    #                 os.path.join(output_dir, 'images'))

    ### GROUP MATCH
    # print('Group match')
    create_order_cluster_heatmap(tax, os.path.join(output_phylo, 'group_match.pdf'))
    create_order_cluster_heatmap_rev(tax, os.path.join(output_phylo, 'group_match_rev.pdf'))
    create_combined_heatmap(tax, os.path.join(output_phylo, 'combined_group_match.pdf'))


    # ### random clusters ###
    mean_phylo_random = []
    consensus_random = []
    print('Computing random...')
    for i in range(N_random):
        print(f'Random {i}')
        random_clusters(tax, cluster_sizes, f'clusters_{i}')
        _, _, mean_phylo_random_i, _ = compute_group_dist(tax, f'clusters_{i}', phylo_dist)
        _, _, consensus_random_i, _ = compute_group_dist(tax, f'clusters_{i}', df_consensus)
        mean_phylo_random.extend(mean_phylo_random_i)
        consensus_random.extend(consensus_random_i)
    
    plot_dist_means(mean_phylo_order, mean_phylo_random, mean_phylo_order_cluster, os.path.join(output_phylo, 'dist_phylo_w_random.pdf'))
    plot_phylo_vs_cons_w_random(mean_phylo_order_cluster, mean_consensus_order_cluster,
                mean_phylo_order, mean_consensus_order,
                mean_phylo_random, consensus_random,
                os.path.join(output_phylo))
    # phylogenetic_plot(mean_phylo_order_cluster, mean_consensus_order_cluster,
    #             mean_phylo_order, mean_consensus_order,
    #             mean_phylo_random, consensus_random,
    #             os.path.join(output_dir, 'images'))


    ### Save tax csv ####
    tax.to_csv(os.path.join(output_dir,'tax_and_clusters.csv'))

    ###### PHYLO COMPARE PLOT #####
    ###### PHYLO COMPARE PLOT #####
    # Define the subplot grid, with extra space between pairs of plots




    #### plot

    # ### TODO: Run clusters matching each taxonomic order
    # k_list = clusters.values.unique()
    # n_species_list = []
    # mean_distance_list = []
    # mean_trait_distance_list = []

    # for k in k_list:
    #     cluster_k = clusters[clusters['cluster'] == k]

    #     # get species from cluster_k (accepted_bin)
    #     species = cluster_k.index
    #     # change spaces to underscores
    #     species_ = species.str.replace(' ', '_')
    #     n_species_k = len(species)

    #     # iterate over species and get the phylogenetic distance
    #     phylo_dist_k = phylo_dist.loc[species_, species_]

    #     # get the mean phylogenetic distance (for non-diagonal elements)    
    #     mean_k = np.sum(phylo_dist_k.values) / (n_species_k*(n_species_k-1))

    #     print(f'Cluster {k}, mean distance {mean_k}, n_species {len(species)}')

    #     n_species_list.append(len(species))
    #     mean_distance_list.append(mean_k)
