#!/usr/bin/env bash

set -euo pipefail

echo "===== Run info ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo

PYTHON="${PYTHON:-.venv/bin/python3}"
PYTHON_CMD="$PYTHON -u"

STEP1="src/cluster_pipeline/step1_cluster_labels.py"
STEP2="src/cluster_pipeline/step2_consensus_matrix.py"
STEP3="src/cluster_pipeline/step3_analyse_consensus.py"

# Experiment settings
SCALES=("True")
RANDOM_ASSIGNS=("False")
SEEDS=$(seq 1 50)

# Controls: set these to true/false depending on what you want to run
RUN_STEP1=false
RUN_STEP2=false
RUN_STEP3=true

# Step 3 options
N_JOBS="${N_JOBS:-10}"
LINKAGE_METHOD="${LINKAGE_METHOD:-ward}"

for scl in "${SCALES[@]}"; do
  for rnd in "${RANDOM_ASSIGNS[@]}"; do
    echo "========================================"
    echo "Running combination: scale=$scl, random_assign=$rnd"
    echo "========================================"

    if [ "$RUN_STEP1" = true ]; then
      echo "--- Step 1: generate labels for all seeds ---"
      for seed in $SEEDS; do
        echo "Running step1 with seed=$seed"
        $PYTHON_CMD "$STEP1" \
          --seed "$seed" \
          --random_assign "$rnd" \
          --scale "$scl"
      done
      echo
    fi

    if [ "$RUN_STEP2" = true ]; then
      echo "--- Step 2: build consensus matrix ---"
      $PYTHON_CMD "$STEP2" \
        --random_assign "$rnd" \
        --scale "$scl"
      echo
    fi

    if [ "$RUN_STEP3" = true ]; then
      echo "--- Step 3: analyse consensus ---"
      $PYTHON_CMD "$STEP3" \
        --random_assign "$rnd" \
        --scale "$scl" \
        --n_jobs "$N_JOBS" \
        --linkage_method "$LINKAGE_METHOD"
      echo
    fi
  done
done

echo "All requested experiments completed."