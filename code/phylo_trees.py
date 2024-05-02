from Bio import Phylo
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = os.path.join(os.getcwd(), 'data', 'phy_tree_BGCI_full.newick')

tree = Phylo.read(data_path, 'newick')

# plt.figure()

# Phylo.draw(tree)

# # Save the figure to a file
# output_path = os.path.join(os.getcwd(), 'output', 'phylo', 'phylo_tree.png')
# plt.savefig(output_path)
# plt.close()  # Close the plot to free up memory

# Compute the distance matrix
def compute_distance_matrix(tree):
    taxa = tree.get_terminals()
    matrix = np.zeros((len(taxa), len(taxa)))
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            matrix[i][j] = tree.distance(taxon1, taxon2)
    return matrix

# Get the distance matrix
dist_matrix = compute_distance_matrix(tree)
print(dist_matrix)