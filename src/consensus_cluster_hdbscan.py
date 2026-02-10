import numpy as np
import pandas as pd
import hdbscan
from functools import partial
import os
from multiprocessing import Pool

def same_cluster_matrix(labels, n):
    """
    Compute the matrix of same cluster membership
    """
    same_cluster = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                same_cluster[i, j] = 1
    return same_cluster

def fit_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(X)
    return clusterer  # Return only the model object

if __name__ == '__main__':

    ### PARAMETERS ###
    verbose = True
    small_data, subset_size = True, 4000
    compute_prob_matrix = True
    n_bootstrap = 10
    n_jobs = 1
    alpha = 0.7 # factor to multiply the error to get the std

    if verbose: 
        print('Running bootstrap with %d bootstrap samples' % n_bootstrap)

    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'data')
    output_path = os.path.join(working_dir, 'output')

    # Load the data
    df_traits = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0) 
    df_errors = pd.read_csv(os.path.join(data_dir, 'pred_errors.csv'), index_col=0)
    if small_data:
        index_list = df_traits.index[:subset_size]
        df_traits = df_traits.loc[index_list,:]
    N_obs = df_traits.shape[0]

    cluster_counts = []
    prob_matrix = np.zeros((N_obs, N_obs))

    bootstrap_dir = os.path.join(output_path, 'consensus_hdbscan')
    if small_data:
        bootstrap_dir = os.path.join(bootstrap_dir, f'small_{subset_size}')
    if alpha != 1:
        bootstrap_dir = os.path.join(bootstrap_dir, f'alpha_{alpha}')
    if not os.path.exists(bootstrap_dir):
        os.makedirs(bootstrap_dir)

    # Loop over bootstrap
    for s in range(n_bootstrap):

        if verbose:
            print('Bootstrap %d' % s)
        
        # resample traits 
        X_s = np.random.normal(df_traits.values, df_errors.values[:,0]*alpha)

        # Fit HDBSCAN
        clusterer = fit_hdbscan(X_s)

        # Store the number of clusters found
        num_clusters = len(np.unique(clusterer.labels_))
        cluster_counts.append(num_clusters)

        # Store cluster membership matrix
        if compute_prob_matrix:
            prob_matrix += same_cluster_matrix(clusterer.labels_, N_obs)

    # Save the number of clusters found in each bootstrap
    df_clusters = pd.DataFrame(cluster_counts, columns=['num_clusters'])
    df_clusters.to_csv(os.path.join(bootstrap_dir, 'cluster_counts.csv'))

    # Compute and save the probability matrix
    if compute_prob_matrix:
        prob_matrix /= n_bootstrap
        df_prob_matrix = pd.DataFrame(prob_matrix, index=df_traits.index, columns=df_traits.index)
        df_prob_matrix.to_csv(os.path.join(bootstrap_dir, 'prob_matrix.csv'))
