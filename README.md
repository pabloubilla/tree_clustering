# Tree Species Clustering for Determining Functional Groups

This repository contains the code for the paper *Functional group
classification using consensus clustering*.\
It implements consensus clustering using resampling of the data with
prediction errors to simulate different scenarios under a clustering
algorithm (here focusing on GMMs).

The resampling scenarios are summarized in a **consensus matrix**, where
component $A_ij$ represents the fraction of times two species were
assigned to the same cluster across runs. A final clustering is obtained
by performing hierarchical clustering on this matrix and selecting the
number of clusters using an optimization metric such as the **silhouette
score**.

**Note:** Some files and datasets are omitted from this repository due
to large file sizes. Please contact the author for access if needed.

------------------------------------------------------------------------

## Repository Structure

    TREE_CLUSTERING/
    ├── data/        # Input datasets, trait data, and metadata
    ├── output/      # Results, consensus matrices, and visualizations
    ├── src/         # Source code for clustering and analysis pipeline

------------------------------------------------------------------------

## How to Run

The clustering pipeline consists of three main steps:

1.  **Generate cluster labels** across multiple seeds\
    `src/cluster_pipeline/step1_cluster_labels.py`

2.  **Build the consensus matrix**\
    `src/cluster_pipeline/step2_consensus_matrix.py`

3.  **Analyse the consensus matrix** with hierarchical clustering\
    `src/cluster_pipeline/step3_analyse_consensus.py`

Each script can be executed individually for debugging or partial runs.

Example:

``` bash
.venv/bin/python3 -u src/cluster_pipeline/step3_analyse_consensus.py   --random_assign False   --scale True   --n_jobs 10   --linkage_method ward
```

### Running the full experiment pipeline

For convenience, the repository includes a helper script:
`src/pipeline_clusters.sh`.

It allows running the full experiment pipeline and controlling parameter
combinations (e.g., `scale`, `random_assign`, seeds) from a single
place.

Run with:

``` bash
chmod +x src/pipeline_clusters.sh
src/pipeline_clusters.sh
```

---

## Download Data

The required dataset can be downloaded from the following link:

https://figshare.com/s/791ffaa164de494f7488  

After downloading, place the files into the `data/` directory.

---

## Processed Data

To prepare the data for the clustering pipeline, run the following script:

```bash
src/prepare_traits.py
```

---

## Contact

For questions, data access, or collaboration inquiries, please contact: pablo.ubilla-pavez@inria.fr
