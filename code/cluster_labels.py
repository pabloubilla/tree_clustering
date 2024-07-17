import argparse
import numpy as np
import pandas as pd
import hdbscan
import os
import pickle

def fit_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(X)
    return clusterer.labels_

def main(random_seed):
    # np.random.seed(random_seed)
    # data_dir = 'path/to/data'  # Update this path
    # output_dir = 'path/to/output'  # Update this path
    
    # df_traits = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0)
    # df_errors = pd.read_csv(os.path.join(data_dir, 'pred_errors.csv'), index_col=0)
    
    # subset_size = 4000  # Adjust as needed
    # alpha = 0.7

    # if df_traits.shape[0] > subset_size:
    #     df_traits = df_traits.iloc[:subset_size]
    #     df_errors = df_errors.iloc[:subset_size]

    # X = np.random.normal(df_traits.values, df_errors.values * alpha)

    np.random.seed(random_seed)
    verbose = True
    small_data, subset_size = False, 10000
    # compute_prob_matrix = True
    # prob_assignation = False
    # n_bootstrap = 10
    # n_component_list = [35,40,45,50,55,60,65,70]
    # n_jobs = 8
    # n_component_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # n_component_list = [2]
    
    # if verbose: 
    #     print('Running bootstrap with %d bootstrap samples' % n_bootstrap)
        # print('Number of components: %s' % n_component_list)

    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'data')
    output_path = os.path.join(working_dir, 'output', 'consensus')
    # create if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # if small data add small_subset to the path
    if small_data:
        output_path = os.path.join(output_path, f'small_{subset_size}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        output_path = os.path.join(output_path, 'full_data')
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    # Load the data
    df_traits_pred = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0) 
    df_traits_obs = pd.read_csv(os.path.join(data_dir, 'traits_obs_log.csv'), index_col=0)
    observed_traits = df_traits_obs.columns
    # df_errors = pd.read_csv(os.path.join(data_dir, 'pred_errors.csv'), index_col=0)
    
    ### Load the error distribution ###
    error_dic = pickle.load(open(os.path.join(data_dir, 'error_pred_dist.pkl'), 'rb'))
    
    if small_data:
        # random_index = np.random.choice(df_traits.index, subset_size, replace=False)
        index_list = df_traits_pred.index[:subset_size]
        df_traits_pred = df_traits_pred.loc[index_list,:]
    N_obs = df_traits_pred.shape[0]

    ### Load the gymnosperm and angiosperm data ###
    gymnosperm = pd.read_csv(os.path.join(data_dir, 'gymnosperms.csv'), index_col=0)['accepted_bin'].values
    angiosperm = pd.read_csv(os.path.join(data_dir, 'angiosperms.csv'), index_col=0)['accepted_bin'].values
    # intersect to df_trait_pred.index
    gymnosperm = np.intersect1d(gymnosperm, df_traits_pred.index)
    angiosperm = np.intersect1d(angiosperm, df_traits_pred.index)
    N_gymnosperm = gymnosperm.shape[0]
    N_angiosperm = angiosperm.shape[0]


    X_s = np.ones(df_traits_pred.shape) * np.nan
    # as a dataframe now
    X_s = pd.DataFrame(X_s, columns=df_traits_pred.columns, index=df_traits_pred.index)

    for index_trait, trait in enumerate(df_traits_pred.columns):

        # Gymnosperm resample
        sampled_error_gym = np.random.choice(error_dic['gymnosperm'][trait], N_gymnosperm, replace=True)
        X_s.loc[gymnosperm, trait] = df_traits_pred.loc[gymnosperm, trait].copy() + sampled_error_gym

        # Angiosperm resample
        sampled_error_ang = np.random.choice(error_dic['angiosperm'][trait], N_angiosperm, replace=True)
        X_s.loc[angiosperm, trait] = df_traits_pred.loc[angiosperm, trait].copy() + sampled_error_ang

        if trait in observed_traits:
            # refill those with complete data
            complete_index = df_traits_obs[trait].dropna().index
            # intersect with df_traits_pred.index   
            complete_index = np.intersect1d(complete_index, df_traits_pred.index)
            X_s.loc[complete_index, trait] = df_traits_pred.loc[complete_index, trait].copy()


    labels = fit_hdbscan(X_s)
    
    # Save labels to a CSV file named by the seed
    pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HDBSCAN for a specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrapping.')
    args = parser.parse_args()

    main(args.seed)
