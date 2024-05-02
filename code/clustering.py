# import necessary libraries
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
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

def histplot_clusters(num_species, output_path, file_name = 'histplot_clusters'):
    '''
    Plot the distribution of the number of species in the clusters and save the plot.
    '''
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Use subplots for better control
    sns.histplot(num_species, ax=ax, color='hotpink', 
                 edgecolor='black')

    # Setting title and labels
    # ax.set_title('Number of Species in Clusters', pad=20)
    ax.set_xlabel('Number of Species', labelpad=10)
    ax.set_ylabel('Frequency', labelpad=10)

    file_path = os.path.join(output_path, file_name) + '.png'

    # Save the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory

def means_pca(cluster_means, weights, output_path, file_name = 'cluster_means'):
    '''
    Plot the cluster means in 2D using PCA and save the plot.
    '''
    # Perform PCA
    pca = PCA(n_components=2)
    cluster_means_pca = pca.fit_transform(cluster_means)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Use subplots for better control
    sns.scatterplot(x=cluster_means_pca[:, 0], y=cluster_means_pca[:, 1], 
                     ax=ax, s=50, hue = weights, markers='D', 
                     linewidth=1, edgecolor='black', palette="viridis")

    # Setting title and labels
    ax.set_title('Cluster Means in 2D', pad=20)
    ax.set_xlabel('PC1', labelpad=10)
    ax.set_ylabel('PC2', labelpad=10)

    file_path = os.path.join(output_path, file_name) + '.png'

    # Save the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory

def dist_pca_plot(df, cluster_means, weights, output_path, file_name = 'dist_pca'):
    '''
    Plot the distribution of the data points in 2D using PCA with focus on clustersand save the plot.
    '''
    # Perform PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    cluster_means_pca = pca.transform(cluster_means)

    num_species = weights*len(df)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Use subplots for better control
    # plot the points very softly
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], ax=ax, 
                    s=20, alpha=0.2, color='grey', edgecolor='none')
    # plot the clusters with big markers
    sns.scatterplot(x=cluster_means_pca[:, 0], y=cluster_means_pca[:, 1],
                     ax=ax, s=50, hue = num_species, markers='D', linewidth=1, edgecolor='black',
                     palette="viridis", legend=False)


    # Color bar setup
    norm = Normalize(vmin=min(num_species), vmax=max(num_species))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Number of Species', fontsize=12, rotation=270, labelpad=15)
    # cbar.set_ticks(np.linspace(min(weights), max(weights), num=5))  # Add custom ticks
    # cbar.set_ticklabels(["Low", "25%", "50%", "75%", "High"])  # Custom labels

    # Setting kabels
    ax.set_xlabel('PC1', labelpad=10)
    ax.set_ylabel('PC2', labelpad=10)

    file_path = os.path.join(output_path, file_name) + '.png'

    # Save the plot
    plt.tight_layout()  
    plt.savefig(file_path, dpi=300)
    plt.close(fig) 

def plot_clusters(df, cluster_labels, output_path, file_name):

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
        centroid = np.mean(df[cluster_indices], axis=0)
        # plot the cluster
        plt.scatter(df[cluster_indices, 0], df[cluster_indices, 1], color=pastel_colors[k],
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
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Clusters {file_name}')
    # save the plot
    plt.savefig(os.path.join(output_path, file_name), dpi=300)
    # reset the plot
    plt.clf()

def run_cluster_methods(cluster_methods, df, output_path):
    '''
    Run the clustering methods and save the results to a csv file
    '''
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
            plot_clusters(df_pca, cluster_labels, output_path, file_name)
            
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
    # df_pca = PCA(n_components=2).fit_transform(df)

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    if parallel:
        # Use Pool to parallelize fitting models if parallel is True
        with Pool() as pool:
            # results = pool.starmap(fit_gmm, [(n, df) for n in n_component_list])
            results = pool.starmap(evaluate_mixture, [(df_train, df_test, n) for n in n_component_list])
            # with tqdm
            # results = list(tqdm.tqdm(pool.starmap(fit_gmm, [(n, df) for n in n_component_list]), total=len(n_component_list)))
    else:
        # Process sequentially if parallel is False
        # results = [fit_gmm(n, df) for n in n_component_list]
        # # with tqdm
        # results = list(tqdm.tqdm((fit_gmm(n, df) for n in n_component_list), total=len(n_component_list)))
        results = [evaluate_mixture(df_train, df_test, n) for n in n_component_list]

    # Extract results
    # n_components, bics = zip(*results)
    bics_train, bics_test, aics_train, aics_test, ll_train, ll_test = zip(*results)

    # Plot the metrics
    plot_metric_list(n_component_list, [bics_train, aics_train, ll_train], 
                     [bics_test, aics_test, ll_test],
                       ['BIC', 'AIC', 'Log-likelihood'], output_path, 
                       ['results_BIC', 'results_AIC', 'results_LL'])
    # plot_metrics(n_component_list, bics_train, 'BIC', output_path, 'BIC_train')
    # plot_metrics(n_component_list, bics_test, 'BIC', output_path, 'BIC_test')
    # plot_metrics(n_component_list, aics_train, 'AIC', output_path, 'AIC_train')
    # plot_metrics(n_component_list, aics_test, 'AIC', output_path, 'AIC_test')
    # plot_metrics(n_component_list, ll_train, 'Log-likelihood', output_path, 'LL_train')
    # plot_metrics(n_component_list, ll_test, 'Log-likelihood', output_path, 'LL_test')

    # save to results to a csv file
    df_results = pd.DataFrame({'n_components': n_component_list, 'BIC_train': bics_train, 'BIC_test': bics_test,
                                'AIC_train': aics_train, 'AIC_test': aics_test, 'LL_train': ll_train, 'LL_test': ll_test})
    df_results.to_csv(os.path.join(output_path, 'gmm_results.csv'), index=False)

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

def plot_metrics(n_component_list, metric_values, metric_name, output_path, file_name):
    '''
    Plot the metric values against the number of components and save the plot.
    '''
    plt.figure()
    plt.plot(n_component_list, metric_values)
    plt.xlabel('Number of components')
    plt.ylabel(metric_name)
    plt.title(f'Gaussian Mixture {metric_name}')
    plt.savefig(os.path.join(output_path, f'GaussianMixture_{file_name}.png'), dpi=300)
    plt.clf()

def evaluate_mixture(df_train, df_test, n_components):

    '''
    Fit a Gaussian Mixture model with n_components components to the training data
    and evaluate it using the BIC, AIC, and log-likelihood on both the training and test data.
    '''
    model = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter=1000,
                            n_init=1, random_state=42)
    model.fit(df_train)
    bic_train, bic_test = model.bic(df_train), model.bic(df_test)
    aic_train, aic_test = model.aic(df_train), model.aic(df_test)
    ll_train = model.score(df_train)
    ll_test = model.score(df_test)
    return bic_train, bic_test, aic_train, aic_test, ll_train, ll_test

def plot_means(cluster_means, trait_labels, output_path, file_name = 'cluster_means', minmax=False):
    
    # Optionally scale the data using Min-Max scaling
    if minmax:
        scaler = MinMaxScaler()
        cluster_means = scaler.fit_transform(cluster_means)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Use subplots for better control
    sns.heatmap(cluster_means, cmap='coolwarm', annot=False, 
                linewidths=.5, ax=ax, cbar_kws={'shrink': .8})

    # Setting title and labels
    ax.set_title('Cluster Means', pad=20)
    ax.set_xlabel('Trait', labelpad=10)
    ax.set_ylabel('Cluster', labelpad=10)

    # Adjusting trait labels to be centered
    ax.set_xticks(range(len(trait_labels)))
    ax.set_xticklabels(trait_labels, rotation=45, ha="center")  # 'ha' is horizontal alignment

    file_path = os.path.join(output_path, file_name) + '_minmax'*minmax + '.png'

    # Save the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(file_path, dpi=300)
    plt.close(fig)  # Close the figure to free memory

def plot_points_2d(df_2d, metric, output_path, file_name = 'points_2d'):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))  # Use subplots for better control
    sns.scatterplot(x=df_2d[:, 0], y=df_2d[:, 1], 
                    hue=metric, palette='coolwarm', ax=ax)

    # Setting title and labels
    # ax.set_title('2D Points', pad=20)
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)

    file_path = os.path.join(output_path, file_name) + '.png'

    # Save the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(file_path, dpi=300)
    plt.close(fig)  # Close the figure to free memory

def samples_mixture(df, n_components, output_path, samples=100):
    # for each sumple shuffle columns
    for s in range(samples):
        df_s = df.apply(lambda x: np.random.permutation(x.values))
        model = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter=1000)
        model.fit(df_s)
        # count how many samples are in each cluster
        cluster_labels = model.predict(df_s)
        df_s['cluster'] = cluster_labels
        cluster_counts = df_s['cluster'].value_counts()
        # plot hist
        histplot_clusters(cluster_counts, output_path, file_name=f'histplot_clusters_sample_{s}')





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
    folder_name = 'complete_data/gaussian_mixture'
    complete = True

    # output path (create if it doesn't exist)
    output_path = os.path.join(os.getcwd(), 'output', folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get path to the data and read 
    # trait_path = os.path.join(os.getcwd(), 'data', 'Estimated_trait_table_with_monos.csv')
    if complete:
        trait_path = os.path.join(os.getcwd(), 'data', 'traits_pred_log.csv')
    else:
        trait_path = os.path.join(os.getcwd(), 'data', 'traits_obs_log.csv')
    # trait_path = os.path.join(os.getcwd(), 'data', 'traits_obs_log.csv')

    df_trait = pd.read_csv(trait_path, index_col=0)
    trait_labels = df_trait.columns
    species_labels = df_trait.index

    # impute missing values
    if not complete:
        imputer = KNNImputer(n_neighbors=30)
        print('Imputing missing values')
        df_trait = imputer.fit_transform(df_trait)

        # save the imputed data
        data_path = os.path.join(os.getcwd(), 'data')
        df_trait_imputed_save = pd.DataFrame(df_trait, columns=trait_labels, index=species_labels)
        df_trait_imputed_save.to_csv(os.path.join(data_path, 'traits_obs_log_imputed.csv'))


    ### TRIAL ###
    K = 100
    model = GaussianMixture(n_components=K, covariance_type='tied', max_iter=1000,
                            n_init=2, random_state=42)
    model.fit(df_trait)

    # some things we can compute
    # bic = model.bic(df_trait)
    # weights = model.weights_s_
    means = model.means_
    weights = model.weights_
    # covariances = model.covariances_
    # scores = model.score_samples(df_trait)
    cluster_labels = model.predict(df_trait)

    df_clusters = pd.DataFrame(df_trait, columns=trait_labels, index=species_labels)
    # round to 1 decimal
    df_clusters = df_clusters.round(1)
    df_clusters['cluster'] = cluster_labels
    # Count the number of observations in each cluster
    cluster_counts = df_clusters['cluster'].value_counts()
    # Create a new column 'cluster_size' using map to associate each 'cluster' with its count
    df_clusters['cluster_size'] = df_clusters['cluster'].map(cluster_counts)
    # Sort the DataFrame first by 'cluster_size' in descending order, then by 'cluster' if you want a secondary sort
    df_clusters = df_clusters.sort_values(by=['cluster_size', 'cluster'], ascending=[False, True])
    # Save the DataFrame to a CSV file
    df_clusters.to_csv(os.path.join(output_path, f'clusters_{K}.csv'))
    # now save one with the means and sizes
    df_means = pd.DataFrame(means, columns=trait_labels)
    # round to 1 decimal
    df_means = df_means.round(1)
    df_means['weight'] = weights
    # df_means['size'] = weights*len(df_trait)
    # round both to 3 decimal
    df_means['weight'] = df_means['weight'].round(3)
    df_means['size'] = cluster_counts
    # order by size
    df_means = df_means.sort_values(by='size', ascending=False)
    df_means.to_csv(os.path.join(output_path, f'cluster_means_{K}.csv'))

    # see how the plot change with shuffling
    samples_mixture(df_trait, K, os.path.join(output_path, 'samples'), samples=100)



    # df_pca = PCA(n_components=2).fit_transform(df_trait)
    # df_umap = umap.UMAP().fit_transform(df_trait)
    # PLOT CLUSTERS
    # plot_clusters(df_umap, cluster_labels, output_path, 'clusters_umap')

    # plot_points_2d(df_umap, scores, output_path, 'scores_ll_umap')

    # plot some stuff
    means_pca(means, weights, output_path, f'cluster_means_pca_k{K}')
    dist_pca_plot(df_trait, means, weights, output_path, f'dist_pca_k{K}')
    plot_means(means, trait_labels, output_path, minmax=True, file_name=f'cluster_means_k{K}')
    histplot_clusters(df_means['size'], output_path, file_name=f'histplot_clusters_k{K}')


    ### FOR RUNNING A RANGE OF COMPONENTS ###
    # n_component_list = [i for i in range(1,20)]
    # run from 10 to 200 in steps of 10
    # n_component_list = [i for i in range(10, 1510, 10)]
    # n_component_list = [i for i in range(10, 50, 10)]
    # # run the clustering methods
    # print('Running Gaussian Mixture')
    # run_mixtures(df_trait, output_path, n_component_list)

 

if __name__ == '__main__':
    main_v2()
