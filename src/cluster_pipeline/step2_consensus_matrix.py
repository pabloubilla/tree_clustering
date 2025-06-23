import pandas as pd
import os
import re
import numpy as np
import timeit
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

def same_cluster_matrix(consensus_matrix, labels):
    # Create an n x n boolean matrix where each element (i, j) is True if labels[i] == labels[j]
    label_matrix = labels[:, None] == labels[None, :]
    # Use the boolean matrix to increment the consensus matrix
    consensus_matrix += label_matrix

def check_if_first(filename):
    # Define a regex pattern to extract the Y value from the filename
    pattern = re.compile(r'labels_seed_\d+_(\d+)\.csv')
    
    match = pattern.search(filename)
    if match:
        y_value = int(match.group(1))
        return y_value == 0
    else:
        return False

def process_files(files):
    N_files = len(files)
    
    # read the first file to get the number of observations
    df = pd.read_csv(files[0], index_col=0)
    N_obs = df.shape[0]
    # initialize the consensus matrix
    consensus_matrix = np.zeros((N_obs, N_obs))
    
    for file in files:
        print(f'Running file {file}')
        # read the labels
        labels = pd.read_csv(file).iloc[:, 0].values
        # change -1 to nan
        labels = np.where(labels == -1, np.nan, labels)
        # add to the consensus matrix
        same_cluster_matrix(consensus_matrix, labels)
    
    consensus_matrix /= N_files  # take average
    return consensus_matrix

def generate_G_list(files):
    G_list = []
    for file in files:
        print(f'Checking file {file}')
        # read the labels
        
        # change -1 to nan
        # labels = np.where(labels == -1, np.nan, labels)
        if check_if_first(file):
            labels = pd.read_csv(file).iloc[:, 0].values
            labels = labels[~np.isnan(labels)]
            print(file, '>>>>>>>>> IT IS FIRST')
            G_list.append(len(np.unique(labels)))
    return G_list

def save_matrix_as_parquet(matrix, output_dir):
    df_matrix = pd.DataFrame(matrix)
    table = pa.Table.from_pandas(df_matrix)
    pq.write_table(table, os.path.join(output_dir, 'consensus_matrix.parquet'))

def plot_G_distribution(G_list, output_file):
    # Convert G_list to a pandas Series
    G_series = pd.Series(G_list)

    print('G LIST')
    print(G_list)
    
    # plot distribution of G with histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(G_series, bins=len(G_series.unique()), discrete=True, color='skyblue')
    
    # x axis
    plt.xlabel('Optimal Number of Groups $(G^{*})$')
    plt.xticks(ticks=sorted(G_series.unique()))  # Set x-ticks to be the unique values in G_list
    
    # y axis
    plt.ylabel('Frequency')
    
    # Save the figure
    plt.savefig(output_file)

    # Show the plot (optional)
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process and generate consensus matrix and G distribution.")
    parser.add_argument('--method', type=str, default='gmm_error1.0_rnd', help='Method name for processing')
    args = parser.parse_args()
    
    method = args.method
    # consensus_data = 'Wood density_Leaf area'
    consensus_data = 'small_100'
    output_dir = os.path.join('output', 'consensus', method, consensus_data)
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    print('Output dir: ', output_dir)
    files = glob.glob(os.path.join(output_dir, 'labels_*'))
    N_files = len(files)
    print(f'Identified {N_files} files')

    # process the files and update the consensus matrix
    t_start = timeit.default_timer()
    consensus_matrix = process_files(files)
    t_end = timeit.default_timer()
    print('Time taken to run the consensus matrix processing: ')
    print(t_end - t_start)

    # generate G list
    t_start = timeit.default_timer()
    G_list = generate_G_list(files)
    t_end = timeit.default_timer()
    print('Time taken to generate G_list: ')
    print(t_end - t_start)

    # plot G distribution
    plot_G_distribution(G_list, os.path.join(image_dir, f'G_dist_{len(G_list)}.pdf'))

    # save the consensus matrix as parquet
    t_start = timeit.default_timer()
    save_matrix_as_parquet(consensus_matrix, output_dir)
    t_end = timeit.default_timer()
    print('Time to save as Parquet (Arrow): ', t_end - t_start)

if __name__ == '__main__':
    main()

















####### OLD CODE #####
# import pandas as pd
# import os
# import re
# import numpy as np
# import timeit
# import pyarrow as pa
# import pyarrow.parquet as pq
# import matplotlib.pyplot as plt
# import seaborn as sns
# import glob
# import sys

# def same_cluster_matrix(consensus_matrix, labels):
#     # Create an n x n boolean matrix where each element (i, j) is True if labels[i] == labels[j]
#     label_matrix = labels[:, None] == labels[None, :]
#     # Use the boolean matrix to increment the consensus matrix
#     consensus_matrix += label_matrix

# def check_if_first(filename):
#     # Define a regex pattern to extract the Y value from the filename
#     pattern = re.compile(r'labels_seed_\d+_(\d+)\.csv')
    
#     match = pattern.search(filename)
#     if match:
#         y_value = int(match.group(1))
#         return y_value == 0
#     else:
#         return False

# def process_files(files, consensus_matrix):
#     N_files = len(files)
#     G_list = []
#     for file in files:
#         print(f'Running file {file}')
#         # read the labels
#         labels = pd.read_csv(file).iloc[:, 0].values
#         # change -1 to nan
#         labels = np.where(labels == -1, np.nan, labels)
#         # add to the consensus matrix
#         same_cluster_matrix(consensus_matrix, labels)
#         if check_if_first(file):
#             print(file, '>>>>>>>>> IT IS FIRST')
#             # add number of clusters
#             G_list.append(len(set(labels)))
#     consensus_matrix /= N_files  # take average
#     return consensus_matrix, G_list

# def save_matrix_as_parquet(matrix, output_dir):
#     df_matrix = pd.DataFrame(matrix)
#     table = pa.Table.from_pandas(df_matrix)
#     pq.write_table(table, os.path.join(output_dir, 'consensus_matrix.parquet'))

# def plot_G_distribution(G_list, output_dir):
#     # plot distribution of G
#     plt.figure(figsize=(8, 6))
#     sns.histplot(G_list, color='skyblue', kde=False)
#     # x axis
#     plt.xlabel('$G$*')
#     # y axis
#     plt.ylabel('Frequency')
#     # savefig
#     plt.savefig(output_dir)

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <method>")
#         sys.exit(1)
    
#     method = sys.argv[1]
#     # consensus_data = 'Wood density_Leaf area'
#     consensus_data = 'full_data'
#     output_dir = os.path.join('output', 'consensus', method, consensus_data)
#     image_dir = os.path.join(output_dir, 'images')
#     os.makedirs(image_dir, exist_ok=True)
#     print('Output dir: ', output_dir)
#     files = glob.glob(os.path.join(output_dir, 'labels_*'))
#     N_files = len(files)
#     print(f'Identified {N_files} files')

#     # read the first file to get the number of observations
#     df = pd.read_csv(files[0], index_col=0)
#     N_obs = df.shape[0]

#     # initialize the consensus matrix
#     consensus_matrix = np.zeros((N_obs, N_obs))

#     # process the files and update the consensus matrix
#     t_start = timeit.default_timer()
#     consensus_matrix, G_list = process_files(files, consensus_matrix)
#     t_end = timeit.default_timer()
#     print('Time taken to run the code v3: ')
#     print(t_end - t_start)

#     # plot G distribution
#     plot_G_distribution(G_list, os.path.join(image_dir, 'G_dist.pdf'))

#     # save the consensus matrix as parquet
#     t_start = timeit.default_timer()
#     save_matrix_as_parquet(consensus_matrix, output_dir)
#     t_end = timeit.default_timer()
#     print('Time to save as Parquet (Arrow): ', t_end - t_start)

# if __name__ == '__main__':
#     main()




# def same_cluster_matrix_v1(consensus_matrix, labels, N_obs):
#     """
#     add to consensus matrix for this iteration
#     """
#     unique_labels = np.unique(labels)
#     # iterate over unique labels
#     for label in unique_labels:
#         # get the indices of the label
#         indices = np.where(labels == label)[0]
#         # iterate over the indices
#         for i in range(len(indices)):
#             for j in range(i+1, len(indices)):
#                 consensus_matrix[indices[i], indices[j]] += 1
#                 consensus_matrix[indices[j], indices[i]] += 1


# def same_cluster_matrix_v2(consensus_matrix, labels, n):
#     """
#     Compute the matrix of same cluster membership
#     """
#     for i in range(n):
#         for j in range(i+1, n):
#             if labels[i] == labels[j]:
#                 consensus_matrix[i, j] += 1
#                 consensus_matrix[j, i] += 1

# def same_cluster_matrix_v3(consensus_matrix, labels):

#     # Create an n x n boolean matrix where each element (i, j) is True if labels[i] == labels[j]
#     label_matrix = labels[:, None] == labels[None, :]

#     # Use the boolean matrix to increment the consensus matrix
#     consensus_matrix += label_matrix



# if __name__ == '__main__':
#     method = 'hdbscan_error0.5'
#     # output_dir
#     output_dir = os.path.join('output', 'consensus', method, 'full_data')
#     # read all files in the output_dir
#     files = glob.glob(os.path.join(output_dir, 'labels_*'))
#     N_files = len(files)
#     print(f'Identified {N_files} files')

#     # read the first file
#     df = pd.read_csv(files[0], index_col=0)
#     # get the number of observations
#     N_obs = df.shape[0]
#     # initialize the consensus matrix
#     consensus_matrix = np.zeros((N_obs, N_obs))
#     # iterate over the files

#     # ### See version 1 ###
#     # t_start = timeit.default_timer()
#     # for file in files:
#     #     # read the labels
#     #     labels = pd.read_csv(os.path.join(output_dir, file)).iloc[:,0].values
#     #     # add to the consensus matrix
#     #     same_cluster_matrix_v1(consensus_matrix, labels, N_obs)
#     # t_end = timeit.default_timer()
#     # print('Time taken to run the code v1: ')
#     # print(t_end - t_start)

#     # ### See version 2 ###
#     # t_start = timeit.default_timer()
#     # for file in files:
#     #     # read the labels
#     #     labels = pd.read_csv(os.path.join(output_dir, file)).iloc[:,0].values
#     #     # add to the consensus matrix
#     #     same_cluster_matrix_v2(consensus_matrix, labels, N_obs)
#     # t_end = timeit.default_timer()
#     # print('Time taken to run the code v2: ')
#     # print(t_end - t_start)

#     ### See version 3 ### (FASTER)
#     t_start = timeit.default_timer()
#     for file in files:
#         print(f'Running file {file}')
#         # read the labels
#         labels = pd.read_csv(file).iloc[:,0].values
#         # change -1 to nan
#         labels = np.where(labels == -1, np.nan, labels)
#         # add to the consensus matrix
#         same_cluster_matrix_v3(consensus_matrix, labels)
#     consensus_matrix = consensus_matrix/N_files # take average
#     t_end = timeit.default_timer()
#     print('Time taken to run the code v3: ')
#     print(t_end - t_start)

#     # TRIANGLE MATRIX
#     # upper_triangle = sp.triu(consensus_matrix, k=1)

#     # ### SPARSE UPPER TRIANGLE ###
#     # t_start = timeit.default_timer()
#     # sparse_matrix = sp.csr_matrix(upper_triangle)
#     # sp.save_npz(os.path.join(output_dir,'consensus_matrix_upper_sparse.npz'), sparse_matrix)
#     # t_end = timeit.default_timer()
#     # print('Time to save sparse (upper triangle): ', t_end - t_start)


#     # ### ARROW UPPER TRIANGLE ###
#     # df_upper_triangle = pd.DataFrame(upper_triangle.toarray())
#     # t_start = timeit.default_timer()
#     # table = pa.Table.from_pandas(df_upper_triangle)
#     # pq.write_table(table, os.path.join(output_dir, 'consensus_matrix_upper.parquet'))
#     # t_end = timeit.default_timer()
#     # print('Time to save as Parquet (upper triangle): ', t_end - t_start)

#     ### ARROW ### (THIS IS PROBABLY THE BEST)
#     df_matrix = pd.DataFrame(consensus_matrix)
#     t_start = timeit.default_timer()
#     table = pa.Table.from_pandas(df_matrix)
#     pq.write_table(table, os.path.join(output_dir, 'consensus_matrix.parquet'))
#     t_end = timeit.default_timer()
#     print('Time to save as Parquet (Arrow): ', t_end - t_start)


#     # ### SAVE AS SPARSE ###
#     # t_start = timeit.default_timer()
#     # # make it sparse
#     # sparse_matrix = sp.csr_matrix(consensus_matrix)
#     # # Save the sparse matrix
#     # sp.save_npz(os.path.join(output_dir,'consensus_matrix_sparse.npz'), sparse_matrix)
#     # t_end = timeit.default_timer()
#     # print('Time to save sparse: ', t_end - t_start)

#     # ### SAVE AS NPY ###
#     # t_start = timeit.default_timer()
#     # # save as npy
#     # print(f'Saving a {N_obs}x{N_obs} matrix')
#     # np.save(os.path.join(output_dir,'consensus_matrix.npy'), consensus_matrix)
#     # t_end = timeit.default_timer()
#     # print('Time to save npy: ', t_end - t_start)


#     # ### SAVE AS CSV ###
#     # t_start = timeit.default_timer()
#     # np.savetxt(os.path.join(output_dir, 'consensus_matrix.csv'), consensus_matrix, delimiter=',')
#     # t_end = timeit.default_timer()
#     # print('Time to save csv: ', t_end - t_start)




#     # sparse: 1362 s, 4.3 Gb
#     # sparse (upper): 544 s, 2 Gb 
#     # npy: 159 s, 18 Gb
#     # csv 573 s, 57 Gb
#     # Parquet (upper): 50 s, 400 Mb


