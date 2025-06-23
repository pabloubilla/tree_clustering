# import necessary libraries
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import umap

import os
import tqdm

from multiprocessing import Pool

def evaluate_mixture(df_train, df_test, n_components, covariance_type='tied'):

    '''
    Fit a Gaussian Mixture model with n_components components to the training data
    and evaluate it using the BIC, AIC, and log-likelihood on both the training and test data.
    '''
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=1000,
                            n_init=1, random_state=42)
    model.fit(df_train)
    bic_train, bic_test = model.bic(df_train), model.bic(df_test)
    aic_train, aic_test = model.aic(df_train), model.aic(df_test)
    ll_train = model.score(df_train)
    ll_test = model.score(df_test)
    return bic_train, bic_test, aic_train, aic_test, ll_train, ll_test

def cross_validate_gmm(df, n_component_list, output_path='', n_splits=5, cov_type='tied', parallel=False, n_jobs = None, verbose=False):
    '''
    Perform cross-validation on the Gaussian Mixture model with different numbers of components.
    '''

    # Split the data into n_splits folds
    n_species = len(df)
    indices_species = np.arange(n_species)
    np.random.shuffle(indices_species)
    fold_size = n_species // n_splits
    fold_indices = [indices_species[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

    # Create a dictionary to store the training and test data for each fold
    train_test_data = {
        i: {
            "train": df.iloc[np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])],
            "test": df.iloc[fold_indices[i]]
        }
        for i in range(n_splits)
    }

    # Initialize lists to store the metrics
    ll_train_list = []
    ll_test_list = []

    # Prepare the arguments for parallel processing
    tasks = []
    for n_components in n_component_list:
        for i in range(n_splits):
            df_train = train_test_data[i]["train"]
            df_test = train_test_data[i]["test"]
            tasks.append((df_train, df_test, n_components, cov_type))

    if parallel:
        # Run the tasks in parallel
        with Pool(n_jobs) as pool:
            results = pool.starmap(evaluate_mixture, tasks)
    else:
        # Run the tasks sequentially
        results = [evaluate_mixture(*task) for task in tasks]

    # Collect the results
    for idx, n_components in enumerate(n_component_list):
        ll_train_splits = []
        ll_test_splits = []
        for i in range(n_splits):
            _, _, _, _, ll_train, ll_test = results[idx * n_splits + i]
            ll_train_splits.append(ll_train)
            ll_test_splits.append(ll_test)
        ll_train_list.append(np.mean(ll_train_splits))
        ll_test_list.append(np.mean(ll_test_splits))

    # Store results in CSV
    df_results = pd.DataFrame({'n_components': n_component_list, 'LL_train': ll_train_list, 'LL_test': ll_test_list})
    df_results.to_csv(os.path.join(output_path, 'gmm_results_cv.csv'), index=False)

    # Plot metrics
    plot_metric_list(n_component_list, [ll_train_list], [ll_test_list], ['Log-likelihood'], output_path, [f'results_LL_cv_{n_splits}'])

def plot_metric_list(n_component_list, metric_train_list, metric_test_list, metric_name_list, output_path, file_names):


    # for each metric plot train and test in same plot
    for metric_train, metric_test, metric_name, file_name in zip(metric_train_list, metric_test_list, metric_name_list, file_names):
        plt.figure()
        plt.plot(n_component_list, metric_train, label='Train', color = 'b')
        plt.plot(n_component_list, metric_test, label='Test', linestyle='--', color='r')
        plt.xlabel('Number of components')
        plt.ylabel(metric_name)
        # plt.title(f'{metric_name}')
        # Adding a legend to help differentiate between lines
        plt.legend(shadow=True, fancybox=True)
        plt.savefig(os.path.join(output_path, f'{file_name}.png'), dpi=300)
        plt.clf()


if __name__ == '__main__':

    folder_name = 'complete_data/gaussian_mixture_full'
    output_path = os.path.join(os.getcwd(), 'output', folder_name)

    df_trait = pd.read_csv('data/traits_pred_log.csv', index_col=0)

    # n_component_list = [i for i in range(25,1025, 25)]
    # n_component_list = [1,2,5,10,50] + [i for i in range(100, 2100, 100)]
    # n_component_list = [2,5,10]
    n_jobs = 4
    n_component_list = [100,200,500,800,1200,1500,1800,2100]
    print('Components to try')
    print(n_component_list)  
    print('Running Gaussian Mixture')
    # run_mixtures(df_trait, output_path, n_component_list, cov_type='full', parallel=True, verbose=True)
    cross_validate_gmm(df_trait, n_component_list, output_path, n_splits=5, cov_type='full',
                       n_jobs = n_jobs, verbose=True)