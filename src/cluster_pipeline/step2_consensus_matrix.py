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

from utils import str_to_bool, make_method_label

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
    df = pd.read_csv(files[0], header=None)
    N_obs = df.shape[0]
    # initialize the consensus matrix
    consensus_matrix = np.zeros((N_obs, N_obs))
    
    for file in files:
        print(f'Running file {file}')
        # read the labels
        labels = pd.read_csv(file, header=None).iloc[:, 0].values
        # PRINT SHAPE
        print(f'Labels shape: {labels.shape}')
        
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

# def save_matrix_as_parquet(matrix, output_dir):
#     df_matrix = pd.DataFrame(matrix)
#     table = pa.Table.from_pandas(df_matrix)
#     pq.write_table(table, os.path.join(output_dir, 'consensus_matrix.parquet'))

def save_matrix_as_parquet(matrix, output_dir):
    table = pa.Table.from_arrays(
        [pa.array(col) for col in matrix.T],
        names=[f"col_{i}" for i in range(matrix.shape[1])]
    )
    pq.write_table(table, os.path.join(output_dir, "consensus_matrix.parquet"))

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
    parser = argparse.ArgumentParser(description='Run clustering with a specified seed and method.')
    parser.add_argument('--method', type=str, default='gmm', help='Clustering method to use.')
    parser.add_argument('--error_weight', type=float, default=1.0, help='Error weight for sampling.')
    parser.add_argument('--random_assign', type = str_to_bool, default = False)
    parser.add_argument('--scale', type = str_to_bool, default = True)
    parser.add_argument('--two_traits', type = str_to_bool, default = False, help='Whether to plot the results using only two traits (wood density and leaf area).')
    parser.add_argument('--consensus_data', type=str, default='full_data', help='Whether to use the full data or only the two traits for the consensus matrix.')
    args = parser.parse_args()


    args = parser.parse_args()
    
    method = args.method
    error_weight = args.error_weight
    random_assign = args.random_assign
    scale = args.scale
    method_label = make_method_label(method, error_weight, random_assign, scale)
    # consensus_data = 'Wood density_Leaf area'
    consensus_data = args.consensus_data
    output_dir = os.path.join('output', 'consensus', method_label, consensus_data)
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    print('Output dir: ', output_dir)
    files = glob.glob(os.path.join(output_dir, 'labels_*'))
    # only take first 10 (CHANGE THIS BACK TO ALL FILES LATER)
    # files = files[:3]
    N_files = len(files)
    print(f'Identified {N_files} files')

    # process the files and update the consensus matrix
    t_start = timeit.default_timer()
    consensus_matrix = process_files(files)
    print('Consensus matrix shape: ', consensus_matrix.shape)
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
    matrix_shape = consensus_matrix.shape
    print(f'Saving consensus matrix of size {matrix_shape} as Parquet (Arrow)...')
    save_matrix_as_parquet(consensus_matrix, output_dir)
    t_end = timeit.default_timer()
    print('Time to save as Parquet (Arrow): ', t_end - t_start)

if __name__ == '__main__':
    main()













