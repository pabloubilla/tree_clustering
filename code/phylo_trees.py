from Bio import Phylo
import matplotlib.pyplot as plt
import numpy as np
import os


# tree = Phylo.read(data_path, 'newick')

# # plt.figure()

# # Phylo.draw(tree)

# # # Save the figure to a file
# # output_path = os.path.join(os.getcwd(), 'output', 'phylo', 'phylo_tree.png')
# # plt.savefig(output_path)
# # plt.close()  # Close the plot to free up memory

# # Compute the distance matrix
# def compute_distance_matrix(tree):
#     taxa = tree.get_terminals()
#     matrix = np.zeros((len(taxa), len(taxa)))
#     for i, taxon1 in enumerate(taxa):
#         for j, taxon2 in enumerate(taxa):
#             matrix[i][j] = tree.distance(taxon1, taxon2)
#     return matrix

# # Get the distance matrix
# dist_matrix = compute_distance_matrix(tree)
# print(dist_matrix)


if __name__ == "__main__":
    # import sys
    
    # if len(sys.argv) < 2:
    #     print('USAGE: python nwk2mat.py TREE.nwk')
    #     sys.exit(1)
    
    import pandas as pd
    import itertools
    from Bio import Phylo
    import tqdm

    data_path = os.path.join(os.getcwd(), 'data', 'phy_tree_BGCI_full.newick')

    # ifile = sys.argv[1]
    
    t = Phylo.read(data_path, 'newick')

    d = {}
    n_terminals = len(t.get_terminals())
    print(f'Computing distances for {n_terminals} terminal nodes')
    for x, y in tqdm.tqdm(itertools.combinations(t.get_terminals(), 2)):
        v = t.distance(x, y)
        d[x.name] = d.get(x.name, {})
        d[x.name][y.name] = v
        d[y.name] = d.get(y.name, {})
        d[y.name][x.name] = v
    for x in t.get_terminals():
        d[x.name][x.name] = 0

    m = pd.DataFrame(d)
    m.to_csv('distances.csv', sep='\t')