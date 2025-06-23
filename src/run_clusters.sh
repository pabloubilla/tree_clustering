#!/bin/bash

# Define directories
script="code/cluster_labels.py"  # Update this path

# Run the clustering script for each seed in parallel
for seed in {1..20}; do
    python3 "$script" --seed "$seed" &
done

# Wait for all background jobs to finish
wait
echo "All clustering processes have completed."

