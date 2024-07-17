import pandas as pd
import os
import numpy as np
import timeit

def same_cluster_matrix_v1(consensus_matrix, labels, N_obs):
    """
    add to consensus matrix for this iteration
    """
    unique_labels = np.unique(labels)
    # iterate over unique labels
    for label in unique_labels:
        # get the indices of the label
        indices = np.where(labels == label)[0]
        # iterate over the indices
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                consensus_matrix[indices[i], indices[j]] += 1
                consensus_matrix[indices[j], indices[i]] += 1


def same_cluster_matrix_v2(consensus_matrix, labels, n):
    """
    Compute the matrix of same cluster membership
    """
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                consensus_matrix[i, j] += 1
                consensus_matrix[j, i] += 1

def same_cluster_matrix_v3(consensus_matrix, labels):

    # Create an n x n boolean matrix where each element (i, j) is True if labels[i] == labels[j]
    label_matrix = labels[:, None] == labels[None, :]

    # Use the boolean matrix to increment the consensus matrix
    consensus_matrix += label_matrix



if __name__ == '__main__':
    # output_dir
    output_dir = os.path.join('output', 'consensus', 'small_120')
    # read all files in the output_dir
    files = os.listdir(output_dir)
    # read the first file
    df = pd.read_csv(os.path.join(output_dir, files[0]), index_col=0)
    # get the number of observations
    N_obs = df.shape[0]
    # initialize the consensus matrix
    consensus_matrix = np.zeros((N_obs, N_obs))
    # iterate over the files

    # ### See version 1 ###
    # t_start = timeit.default_timer()
    # for file in files:
    #     # read the labels
    #     labels = pd.read_csv(os.path.join(output_dir, file)).iloc[:,0].values
    #     # add to the consensus matrix
    #     same_cluster_matrix_v1(consensus_matrix, labels, N_obs)
    # t_end = timeit.default_timer()
    # print('Time taken to run the code v1: ')
    # print(t_end - t_start)

    # ### See version 2 ###
    # t_start = timeit.default_timer()
    # for file in files:
    #     # read the labels
    #     labels = pd.read_csv(os.path.join(output_dir, file)).iloc[:,0].values
    #     # add to the consensus matrix
    #     same_cluster_matrix_v2(consensus_matrix, labels, N_obs)
    # t_end = timeit.default_timer()
    # print('Time taken to run the code v2: ')
    # print(t_end - t_start)

    ### See version 3 ###
    t_start = timeit.default_timer()
    for file in files:
        # read the labels
        labels = pd.read_csv(os.path.join(output_dir, file)).iloc[:,0].values
        # change -1 to nan
        labels = np.where(labels == -1, np.nan, labels)
        # add to the consensus matrix
        same_cluster_matrix_v3(consensus_matrix, labels)
    t_end = timeit.default_timer()
    print('Time taken to run the code v3: ')
    print(t_end - t_start)
    # save the consensus matrix as csv
    np.savetxt('consensus_matrix.csv', consensus_matrix, delimiter=',')