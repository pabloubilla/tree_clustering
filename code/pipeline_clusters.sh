#!/bin/bash

# Define directories
cluster_script="code/cluster_labels.py"  # Update this path
consensus_script="code/consensus_matrix.py"  # Path to consensus matrix script
analyse_script="code/analyse_consensus.py"  # Path to analyse consensus script

# Define method and number of components for GMM
method="hdbscan"  # Options: 'hdbscan' or 'gmm'
# components=20  # Specify number of components if using GMM
# rnd="_rnd" # If using rnd assignation
rnd=""
scl=""
random_assign="False"
scale="False"

# Set the number of threads for MKL, OpenBLAS, and OMP (used by NumPy, SciPy)
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Define a list of error weights to iterate over
# error_weights=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
error_weights=(1.0)

for error_weight in "${error_weights[@]}"; do
    # Construct the method string with error weight
    method_with_error="${method}_error${error_weight}"

    # Run the clustering script for each seed in parallel
    for seed in {1..50}; do
        /home/pablo/.venv/bin/python3 "$cluster_script" --seed "$seed" --method "$method" --random_assign "$random_assign" --scale "$scale" &
    done

    # Wait for all background jobs to finish
    wait
    echo "All clustering processes have completed for error_weight=${error_weight}."

    # Run consensus matrix script
    /home/pablo/.venv/bin/python3 "$consensus_script" --method "$method_with_error""$rnd""$scl"
    echo "Consensus matrix computation completed for error_weight=${error_weight}."

    # Run analyse consensus script
    /home/pablo/.venv/bin/python3 "$analyse_script" "$method_with_error""$rnd""$scl" 20
    echo "Analysis of consensus matrix completed for error_weight=${error_weight}."
done

echo "All processes have completed for all error weights."