import pandas as pd
import numpy as np
import os
from matplotlib.colors import to_hex, ListedColormap
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
from mpl_toolkits.mplot3d import Axes3D

def linewidth():
    return 6.30045


def load_clusters(output_dir):
    cluster_path = os.path.join(output_dir, 'final_clusters.csv')
    print(f"Cluster path: {cluster_path}")  # Debugging statement
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"File not found: {cluster_path}")
    return pd.read_csv(cluster_path, header=None).values.flatten()

def load_original_data(data_path):
    print(f"Loading original data from {data_path}")
    return pd.read_csv(data_path, index_col=0)


def create_custom_palette(n_colors):
    # Combine multiple palettes to ensure a diverse set of colors
    palettes = []
    palettes += sns.color_palette("pastel", n_colors=n_colors)
    # palettes += sns.color_palette("husl", n_colors=n_colors)
    # palettes += sns.color_palette("dark", n_colors=n_colors)
    palettes += sns.color_palette("Set2", n_colors=n_colors)
    palettes += sns.color_palette("Set3", n_colors=n_colors)
    # palettes += sns.color_palette("Paired", n_colors=n_colors)
    
    np.random.seed(5)
    # Shuffle the palette list to maximize color contrast
    np.random.shuffle(palettes)
    
    # # Truncate or extend the list to match the requested number of colors
    # if len(palettes) < n_colors:
    #     palettes = palettes * (n_colors // len(palettes) + 1)
    # palettes = palettes[:n_colors]

    return palettes


def plot_pca(data, clusters, images_dir):
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data)
    plt.figure(figsize=(16, 10))
    markers = ['o', 's', 'D']  # Different markers for variety
    palette = create_custom_palette(n_colors=30)
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, 
                    palette=palette, style=clusters, markers=markers, legend=None, s=10)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'pca_clusters.png'))
    plt.clf()

# def plot_tsne(data, clusters, images_dir):
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_components = tsne.fit_transform(data)
#     plt.figure(figsize=(16, 10))
#     markers = ['o', 's', 'D']  # Different markers for variety
#     palette = create_custom_palette(n_colors=30)
#     sns.scatterplot(x=tsne_components[:, 0], y=tsne_components[:, 1], hue=clusters, palette=palette, style=clusters, markers=markers, legend=None, s=10)
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.tight_layout()
#     plt.savefig(os.path.join(images_dir, 'tsne_clusters.png'))
#     plt.clf()


def plot_tsne(data, tsne_components, clusters, images_dir, color_list):

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2'])
    df['cluster'] = clusters

    plt.figure(figsize=(linewidth()*0.85, linewidth()/2.6))
    # markers = ['o', 's', 'D']  # Different markers for variety
    markers = ['.','*','P'] 
    # palette = create_custom_palette(n_colors=len(np.unique(clusters)))
    palette = np.array(color_list)

    
    # Scatter plot
    sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', palette=palette, style='cluster', markers=markers, legend=None, data=df, s=3, alpha = 0.8)

    # Calculate and plot centroids
    # centroids = df.groupby('cluster')[['tsne1', 'tsne2']].mean().reset_index()
    # for ix, row in centroids.iterrows():
    #     plt.text(row['tsne1'], row['tsne2'], str(int(row['cluster'])), 
    #             color='black', 
    #             fontsize=3, weight='bold', ha='center', va='center', 
    #             bbox=dict(facecolor=palette[ix], alpha=0.5, edgecolor='none', pad=0.8))

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(pad = 0.1)
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_numbered.png'), dpi = 400)
    plt.clf()


def plot_tsne_big(data, tsne_components, clusters, images_dir, color_list):

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2'])
    df['cluster'] = clusters

    plt.figure(figsize=(linewidth(), linewidth()))
    # markers = ['o', 's', 'D']  # Different markers for variety
    markers = ['.','*','P'] 
    # palette = create_custom_palette(n_colors=len(np.unique(clusters)))


    # 'darkgoldenrod',
    #     'peru', 'slateblue', 'maroon', 'lightblue'
    palette = np.array(color_list)

    # Scatter plot
    sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', palette=palette, style='cluster', markers=markers, legend=None, data=df, s=5, alpha = 0.8)

    # Calculate and plot centroids
    centroids = df.groupby('cluster')[['tsne1', 'tsne2']].mean().reset_index()
    for ix, row in centroids.iterrows():
        plt.text(row['tsne1'], row['tsne2'], str(int(row['cluster'])), 
                color='black', 
                fontsize=8, weight='bold', ha='center', va='center', 
                bbox=dict(facecolor=palette[ix], alpha=0.5, edgecolor='none', pad=0.8))

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(pad = 0.1)
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_numbered_big.png'), dpi = 400)
    plt.clf()



def plot_tsne_with_numbers(data, tsne_components, clusters, images_dir):

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2'])
    df['cluster'] = clusters

    plt.figure(figsize=(linewidth()*0.85, linewidth()/2.6))
    
    # Create a color palette with as many colors as there are clusters
    palette = create_custom_palette(n_colors=len(np.unique(clusters)))

    # Plot each point as the cluster number with a distinct color
    for cluster in np.unique(clusters):
        subset = df[df['cluster'] == cluster]
        for _, row in subset.iterrows():
            plt.text(row['tsne1'], row['tsne2'], str(int(row['cluster'])),
                     color=palette[cluster],
                     fontsize=3, ha='center', va='center')

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_numberscatter.png'), dpi=400)
    plt.clf()

def plot_tsne_3d_with_perspectives(data, clusters, images_dir):
    tsne = TSNE(n_components=3, random_state=42)
    tsne_components = tsne.fit_transform(data)

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2', 'tsne3'])
    df['cluster'] = clusters

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    palette = create_custom_palette(n_colors=len(np.unique(clusters)))

    for cluster in np.unique(clusters):
        cluster_data = df[df['cluster'] == cluster]
        ax.scatter(cluster_data['tsne1'], cluster_data['tsne2'], cluster_data['tsne3'],
                   s=10, alpha = 0.7, c=np.array(palette[cluster]).reshape(1, -1), label=f'Cluster {cluster}')

    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)
    ax.set_zlabel('t-SNE 3', fontsize=14)

    # Save the first perspective
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_3d_perspective1.png'))

    # Rotate and save the second perspective
    ax.view_init(elev=20., azim=45)  # Adjust these parameters as needed
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_3d_perspective2.png'))

    # Rotate and save the third perspective
    ax.view_init(elev=20., azim=90)  # Adjust these parameters as needed
    plt.savefig(os.path.join(images_dir, 'tsne_clusters_3d_perspective3.png'))

    plt.clf()

def plot_traits(data, clusters, traits, images_dir):
    plt.figure(figsize=(10,8))
    markers = ['o']  # Different markers for variety
    palette = create_custom_palette(n_colors=10)
    sns.scatterplot(x=data[traits[0]], y=data[traits[1]],
    hue=clusters, palette=palette, style=clusters, markers=markers, legend=None, s=10)
    plt.xlabel(traits[0])
    plt.ylabel(traits[1])
    plt.savefig(os.path.join(images_dir, 'final_cluster_traits.png'))
    plt.clf()

def plot_cluster_distribution(clusters, images_dir):
    bin_intervals = 100
    cluster_sizes = pd.Series(clusters).value_counts().values
    
    # Calculate mean and standard deviation
    mean_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    
    # Print mean and standard deviation
    print(f'Cluster Size Distribution - Mean: {mean_size:.2f}, Std Dev: {std_size:.2f}')
    
    plt.figure(figsize=(linewidth()*0.48, linewidth()*0.48/1.8))
    sns.histplot(cluster_sizes, bins=range(0, max(cluster_sizes) + bin_intervals, bin_intervals),
                 kde=False, color='skyblue', edgecolor='black', linewidth=1)
    plt.xlabel('Cluster size')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(images_dir, 'cluster_size_distribution.pdf'))
    plt.clf()


def plot_consensus_histogram(df_clusters, output_file):
    bins = [i * 0.1 for i in range(11)]
    
    # Calculate mean and standard deviation
    mean_consensus = np.mean(df_clusters['Avg Consensus'])
    std_consensus = np.std(df_clusters['Avg Consensus'])
    
    # Print mean and standard deviation
    print(f'Consensus Histogram - Mean: {mean_consensus:.2f}, Std Dev: {std_consensus:.2f}')
    
    plt.figure(figsize=(linewidth()*0.48, linewidth()*0.48/1.8))
    sns.histplot(df_clusters['Avg Consensus'], bins=bins, kde=False, color='skyblue', edgecolor='black', linewidth=1)
    plt.xlabel('Average consensus')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=0.1)
    plt.savefig(output_file)
    plt.close()

def plot_correlation_heatmap(data, images_dir):
    plt.figure(figsize=(linewidth(), linewidth()))
    
    # Rename columns using the shortened trait names
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
    
    data_renamed = data.rename(columns=shortened_trait_names)
    
    # Calculate the correlation matrix
    corr_matrix = data_renamed.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Use a different diverging colormap (e.g., "coolwarm")
    cmap = sns.diverging_palette(10, 220, sep=80, as_cmap=True)
    
    # Plot the heatmap
    sns.heatmap(corr_matrix, mask=np.eye(len(corr_matrix), dtype=bool), cmap=cmap, center=0, annot=True, 
                fmt=".2f", annot_kws={"fontsize": 6}, linewidths=.5, cbar_kws={"shrink": .5}, vmax = .6, vmin=-.6,
                cbar=False)
    
    # plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'correlation_heatmap.pdf'))
    plt.clf()


def main():

    color_list = [
        'olive', 'salmon', 'lightcoral', 'magenta', 'blue', 'lightsalmon', 'darkred', 'lightgreen', 'darkblue',
        'green', 'darkgreen', 'deepskyblue', 'orange', 'indigo', 'darkorange', 'lightblue', 'purple',
        'lightseagreen', 'pink', 'teal', 'peru', 'plum', 'black', 'sandybrown', 'darkmagenta', 'lime',
        'brown', 'lightgreen', 'coral', 'darkcyan', 'khaki', 'darkviolet', 'violet', 'mediumseagreen',
        'tomato', 'gray', 'gold', 'salmon', 'orchid', 'yellow', 'turquoise', 'tan']

    # method = 'gmm_error1.0_rnd'  # Use the actual method name
    method = sys.argv[1]
    consensus_data = 'full_data'
    # consensus_data = 'Wood density_Leaf area'
    output_dir = os.path.join('output', 'consensus', method, consensus_data)
    data_path = os.path.join('data', 'traits_pred_log.csv')
    
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")
    
    clusters = load_clusters(output_dir)
    data = load_original_data(data_path)
    df_clusters = pd.read_csv(os.path.join(output_dir, 'clusters_info.csv'))
    print(df_clusters.head())

    # plot_traits(data, clusters, ['Wood density', 'Leaf area'], os.path.join(output_dir, 'images'))

    images_dir = os.path.join(output_dir, 'images', 'cluster_results')
    os.makedirs(images_dir, exist_ok=True)
    plot_correlation_heatmap(data, images_dir) # see correlation between traits
    plot_cluster_distribution(clusters, images_dir)
    plot_consensus_histogram(df_clusters, os.path.join(images_dir, 'consensus_histogram.pdf'))


    # plot_pca(data, clusters, images_dir)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(data)
    plot_tsne(data, tsne_components, clusters, images_dir, color_list)
    plot_tsne_big(data, tsne_components, clusters, images_dir, color_list)
    # plot_tsne_with_numbers(data, tsne_components, clusters, images_dir)
    # plot_tsne_3d_with_perspectives(data, clusters, images_dir)

if __name__ == '__main__':
    main()
