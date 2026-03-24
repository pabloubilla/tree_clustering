import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform




# Set random seed for reproducibility
np.random.seed(42)

# Number of plots to generate
N = 3
n_components_list = [3,2,4]

# Number of samples
n_samples = 50

# Original data generation parameters
mean_params = [
    ([5, 0], [[1, 0.5], [0.5, 1]]),
    ([7, 5], [[1, -0.5], [-0.5, 1]]),
    ([8, 0], [[2, 0.5], [0.5, 2]])
]

# Custom colormap for clusters
# colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
# pastel colors
colors = ['coral', 'seagreen', 'cornflowerblue', 'mediumorchid']
cmap = ListedColormap(colors)

# Function to draw ellipses
def draw_ellipse(position, covariance, color, ax=None, **kwargs):
    ax = ax or plt.gca()

    
    # Convert covariance matrix to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, color=color, alpha=0.3, **kwargs))

# Create output directory if it doesn't exist
output_dir = 'diagram/'


for plot_num in range(N):
    # Generate synthetic data
    X = np.vstack([
        np.random.multivariate_normal(mean, cov, n_samples)
        for mean, cov in mean_params
    ])

    # Add a small error term to each data point
    error_term = np.random.normal(scale=0.4, size=X.shape)
    X_noisy = X + error_term

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components_list[plot_num], covariance_type='full')
    gmm.fit(X_noisy)
    labels = gmm.predict(X_noisy)

    # Plot data and ellipses
    plt.figure(figsize=(3, 3))
    # scatter = plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=labels, s=10, cmap=cmap)

    # Plot GMM components with matching colors
    for i, (pos, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        scatter = plt.scatter(X_noisy[labels == i, 0], X_noisy[labels == i, 1], s=10, color=colors[i], alpha = 0.8)
        draw_ellipse(pos, covar, color=colors[i])

    # Label axes
    # plt.xlabel("Trait 1")
    # plt.ylabel("Trait 2")

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    # Save plot to file
    plt.savefig(f'{output_dir}/GMM_plot_{plot_num + 1}.pdf')
    plt.close()

print(f"{N} plots have been saved to the '{output_dir}' directory.")




###### CONSENSUS MATRIX

matrix_size = 99

# Number of groups
n_groups = 3

# Indices for the groups (3 random sizes that add up to matrix_size)
group_sizes = [15, 30, 54]
indices = np.arange(matrix_size)

# Shuffle indices to randomize group membership
np.random.shuffle(indices)

# Initialize consensus matrix with low values
consensus_matrix = np.random.uniform(low=0.05, high=0.35, size=(matrix_size, matrix_size))

# Fill in the higher values for members of the same group
current_idx = 0
for i in range(n_groups):
    start = current_idx
    end = current_idx + group_sizes[i]
    current_idx = end
    group_indices = indices[start:end]
    print(group_indices)
    for idx1 in group_indices:
        for idx2 in group_indices:
            consensus_matrix[idx1, idx2] = np.random.uniform(low=0.65, high=0.9)
            consensus_matrix[idx2, idx1] = consensus_matrix[idx1, idx2]
# diagonal is 1
np.fill_diagonal(consensus_matrix, 1)
# make sure it is symmetric
consensus_matrix = np.maximum(consensus_matrix, consensus_matrix.T)



# Plot the consensus matrix
plt.figure(figsize=(3, 3))
with plt.rc_context({'lines.linewidth': 0.5}):
    plt.imshow(consensus_matrix, cmap='Reds', interpolation='nearest')
# plt.colorbar(label='Consensus Value')
# plt.title('Consensus Matrix')
plt.xticks([])
plt.yticks([])

plt.savefig(f'{output_dir}/consensus_matrix.pdf')
# plt.show()


# Number of groups
n_groups = 3



# Perform hierarchical clustering
# Convert the consensus matrix to a distance matrix
distance_matrix = 1 - consensus_matrix
# diagonal to zero
np.fill_diagonal(distance_matrix, 0)

# Perform hierarchical clustering using 'average' linkage
linkage_matrix = linkage(squareform(distance_matrix), method='ward')

# Get cluster assignments
clusters = fcluster(linkage_matrix, t=n_groups, criterion='maxclust')

# Reorder the consensus matrix based on cluster assignments
sorted_indices = np.argsort(clusters)
sorted_matrix = consensus_matrix[sorted_indices, :][:, sorted_indices]

# Plot the sorted consensus matrix
plt.figure(figsize=(3, 3))
with plt.rc_context({'lines.linewidth': 0.5}):
    plt.imshow(sorted_matrix, cmap='Reds', interpolation='nearest')
plt.xticks([])
plt.yticks([])


plt.savefig(f'{output_dir}/clustered_consensus_matrix.pdf')
# plt.show()

# Plot the dendrogram for the hierarchical clustering
plt.figure(figsize=(10, 6))
# linewidth
with plt.rc_context({'lines.linewidth': 2}):
    dendrogram(linkage_matrix, labels=clusters[sorted_indices], leaf_rotation=90, leaf_font_size=10)
# no ticks
plt.xticks([])
plt.yticks([])
# no text
plt.title('')
plt.xlabel('')
plt.ylabel('')
# no frame
plt.box(False)

# Save the dendrogram plot
plt.savefig(f'{output_dir}/dendrogram.pdf')
# plt.show()