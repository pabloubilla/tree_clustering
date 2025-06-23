# Tree Species Clustering for Determining Functional Groups

This repository contains the code and structure for my MSc dissertation at University College London (UCL), as part of the MSc in Ecology and Data Science. The project investigates the clustering of tree species into functional groups based on trait data. The aim is to explore ecological diversity patterns and determine how many distinct "types" of trees exist, based on their traits.

**Note:** Some files and datasets are omitted from this repository due to large file sizes. Please contact the author for access if needed.

This project is part of an ongoing scientific publication based on my MSc thesis work.

---

## Repository Structure

```
TREE_CLUSTERING/
├── archive/     # Archive of previous runs or experimental outputs
├── data/        # Input datasets, trait data, and metadata
├── output/      # Results, consensus matrices, and visualizations
├── src/         # Source code for clustering and analysis pipeline
```

---

## How to Run

The full clustering pipeline is executed using a bash script (`run_clustering.sh`). This script automates the following steps:

1. Runs clustering across multiple seeds and saves labels.
2. Computes a consensus matrix from the results.
3. Uses hierarchical clustering and silhouette score to determine the optimal aggregation

To execute the pipeline, run:

```bash
bash run_clustering.sh
```

Make sure to check the script and adjust parameters such as the clustering method (`gmm` or `hdbscan`), number of components, scaling options, and error weights as needed.

---

## Contact

For questions, data access, or collaboration inquiries, please contact:

**Pablo Ubilla**  
Email: pabloubilla@ug.uchile.cl
