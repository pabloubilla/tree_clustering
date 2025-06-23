setwd("C:/Users/pablo/OneDrive/Desktop/tree_clustering")

library(ape)
tree <- read.tree(file="data/phy_tree_BGCI_full.newick")
par(mfrow=c(1,1), mar=c(0.1, 0.1, 0.1, 0.1))  # Reset margins and layout
# plot(tree)

# Compute the distance matrix
dist_matrix <- cophenetic(tree)
