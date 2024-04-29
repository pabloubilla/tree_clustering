# import necessary libraries
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

import os
import tqdm

from multiprocessing import Pool

def plot_clusters_PCA(df_pca, cluster_labels, output_path, file_name):

    #### CHECK LABELS, NOT WORKING PROPERLY ####

    # get the number of clusters
    K = len(np.unique(cluster_labels))
    # set the colors for the clusters
    pastel_colors = sns.color_palette("pastel", K)

    # loop through the clusters 
    for k in range(K):
        # get the indices for the cluster
        cluster_indices = cluster_labels == k
        # get centroid of the cluster
        centroid = np.mean(df_pca[cluster_indices], axis=0)
        # plot the cluster
        plt.scatter(df_pca[cluster_indices, 0], df_pca[cluster_indices, 1], color=pastel_colors[k],
                    alpha=0.6, s=5, edgecolors='w', linewidths=0.2)

        dark_color = np.array(pastel_colors[k]) * 0.85
        color_values_dark = tuple(dark_color)

        # plt.annotate(k, 
        #          centroid,
        #          horizontalalignment='center',
        #          verticalalignment='center',
        #          size=7, weight='bold',
        #          color='black',
        #          bbox=dict(facecolor = color_values_dark , 
        #                    edgecolor='black',
        #                      boxstyle='round,pad=0.5', 
        #                      linewidth=0.8))
    # add axis
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'Clusters {file_name}')
    # save the plot
    plt.savefig(os.path.join(output_path, file_name))
    # reset the plot
    plt.clf()

def run_cluster_methods(cluster_methods, df, output_path):

    # get PCA for plotting the clusters
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)

    results = []
    # loop through the methods and parameters   
    for method, params in cluster_methods.items():
        for p_combination in (dict(zip(params, x)) for x in itertools.product(*params.values())):
            # create the model
            model = eval(method)(**p_combination)
            # fit the model and get the cluster labels
            cluster_labels = model.fit_predict(df)
            # calculate the silhouette score
            silhouette_score_ = silhouette_score(df, cluster_labels)
            # save the results for this combination
            results.append({
                    "Algorithm": method,
                    "Parameters": p_combination,
                    "Silhouette Score": silhouette_score_
                })
            # plot the clusters
            file_name = f'{method}_{"_".join([f"{k}_{v}" for k,v in p_combination.items()])}.png'
            plot_clusters_PCA(df_pca, cluster_labels, output_path, file_name)
            
    # save the results to a dataframe
    df_results = pd.DataFrame(results)
    # save the results to a csv file
    df_results.to_csv(os.path.join(output_path, 'clustering_results.csv'), index=False)


def fit_gmm(n_components, df):
    # print(f'Running Gaussian Mixture with {n_components} components')
    model = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter=1000)
    model.fit(df)
    bic = model.bic(df)
    return n_components, bic

def run_mixtures(df, output_path, n_component_list, parallel=True, verbose=False):
    if verbose: print("Starting the mixture model fitting process...")

    # Perform PCA
    df_pca = PCA(n_components=2).fit_transform(df)

    if parallel:
        # Use Pool to parallelize fitting models if parallel is True
        with Pool() as pool:
            results = pool.starmap(fit_gmm, [(n, df) for n in n_component_list])
            # with tqdm
            # results = list(tqdm.tqdm(pool.starmap(fit_gmm, [(n, df) for n in n_component_list]), total=len(n_component_list)))
    else:
        # Process sequentially if parallel is False
        # results = [fit_gmm(n, df) for n in n_component_list]
        # # with tqdm
        results = list(tqdm.tqdm((fit_gmm(n, df) for n in n_component_list), total=len(n_component_list)))

    # Extract results
    n_components, bics = zip(*results)

    # Save BIC values to a CSV file
    df_bic = pd.DataFrame({'n_components': n_component_list, 'bic': bics})
    df_bic.to_csv(os.path.join(output_path, 'GaussianMixture_bic_v2.csv'), index=False)
    
    # Plot BIC values
    plt.figure()
    plt.plot(n_components, bics)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Gaussian Mixture BIC')
    n_low = np.min(n_components)
    n_high = np.max(n_components)
    plt.savefig(os.path.join(output_path, f'GaussianMixture_bic_{n_low}_{n_high}.png'))
    plt.clf()



def main():

    # output path (create if it doesn't exist)
    output_path = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get path to the data and read 
    # trait_path = os.path.join(os.getcwd(), 'data', 'Estimated_trait_table_with_monos.csv')
    trait_path = os.path.join(os.getcwd(), 'data', 'Maynard_et_al_Tree_species_trait_table_with_monos.csv')
    df_trait = pd.read_csv(trait_path)
    df_trait = df_trait.pivot(index='accepted_bin', columns='TraitID', values='value')
    # take a random sample of the data
    df_trait = df_trait.sample(frac=0.05, random_state=42)

    # define the clustering methods to try
    cluster_methods = {
        'KMeans': {'n_clusters': [10,20]},
        'AgglomerativeClustering': {'n_clusters': [5, 10, 20], 'linkage': ['ward', 'complete']},
        # 'DBSCAN': {'eps': [0.1, 0.2, 0.3], 'min_samples': [5, 10]},
        'HDBSCAN': {'min_samples': [5, 10],
                    'cluster_selection_epsilon': [0.1, 0.2]}
    }

    # run the clustering methods
    run_cluster_methods(cluster_methods, df_trait, output_path)

# second version of the main function for Gaussian Mixture
def main_v2():
    print('Running main_v2')

    # output path (create if it doesn't exist)
    output_path = os.path.join(os.getcwd(), 'output', 'gaussian_mixture')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get path to the data and read 
    # trait_path = os.path.join(os.getcwd(), 'data', 'Estimated_trait_table_with_monos.csv')
    trait_path = os.path.join(os.getcwd(), 'data', 'traits_obs_log.csv')

    df_trait = pd.read_csv(trait_path, index_col=0)

    # impute missing values
    imputer = KNNImputer(n_neighbors=30)
    print('Imputing missing values')
    df_trait = imputer.fit_transform(df_trait)

    # trial
    # model = GaussianMixture(n_components=100, covariance_type='tied', max_iter=1000)
    # model.fit(df_trait)
    # bic = model.bic(df_trait)

    # weights = model.weights_
    # print(weights)

    # means = model.means_
    # covariances = model.covariances_


    n_component_list = [i for i in range(1,20)]

    # run the clustering methods
    run_mixtures(df_trait, output_path, n_component_list)

 

if __name__ == '__main__':
    main_v2()
