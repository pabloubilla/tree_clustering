#%%
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
# import umap

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

def run_mixtures(df, output_path, n_component_list, cov_type = 'tied', parallel=True, verbose=False):
    if verbose: print("Starting the mixture model fitting process...")

    # Perform PCA
    # df_pca = PCA(n_components=2).fit_transform(df)

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    if parallel:
        # Use Pool to parallelize fitting models if parallel is True
        with Pool() as pool:
            # results = pool.starmap(fit_gmm, [(n, df) for n in n_component_list])
            results = pool.starmap(evaluate_mixture, [(df_train, df_test, n, cov_type) for n in n_component_list])
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

def cross_validate_gmm(df, n_component_list, output_path='', n_splits=5, cov_type='tied', parallel=False, verbose=False):
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
        with Pool() as pool:
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

def grid_histplot_clusters(df, n_component_list, output_path, file_name = 'grid_histplot_clusters'):
    num_plots = len(n_component_list)
    # Create the with 2 plots per row
    fig, axs = plt.subplots(num_plots//2, 2, figsize=(20, 10*num_plots//2),
                            sharey=True, sharex=True)  # Use subplots for better control
    
    for i, n_components in enumerate(n_component_list):
        print(f'Running Gaussian Mixture with {n_components} components')
        # fit Gussian Mixture
        model = GaussianMixture(n_components=n_components, covariance_type='tied', max_iter=1000)
        model.fit(df)
        cluster_labels = model.predict(df)
        df['cluster'] = cluster_labels
        cluster_counts = df['cluster'].value_counts()
        # Create the plot
        sns.histplot(cluster_counts, ax=axs[i//2, i%2], color='hotpink', 
                     edgecolor='black', binwidth=20)
        # show number of clusters in box
        axs[i//2, i%2].text(0.5, 0.9, f'K={n_components}', horizontalalignment='center',
                            verticalalignment='center', transform=axs[i//2, i%2].transAxes, fontsize=12)
        
        # Setting axis names
        axs[i//2, i%2].set_xlabel('Number of Species')
        axs[i//2, i%2].set_ylabel('Frequency')

    file_path = os.path.join(output_path, file_name) + '.png'
    # save the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory

class BootstrapGMM:
    def __init__(self, n_components=1, max_iter=1000, covariance_type='tied'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.ll_samples = []
        self.mean_samples = []
        self.cov_samples = []

    def fit(self, df, samples=100):
        for s in range(samples):
            df_s = df.sample(frac=1, replace=True)
            model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, max_iter=self.max_iter)
            model.fit(df_s)
            self.ll_samples.append(model.score(df_s))
            self.mean_samples.append(model.means_)
            self.cov_samples.append(model.covariances_)
    
    def save_results(self, output_path):
        results_dict = {
            'll_samples': self.ll_samples,
            'mean_samples': self.mean_samples,
            'cov_samples': self.cov_samples
        }
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(output_path, index=False)

def bootstrap_test(df, g0, g1, cov_type = 'tied', samples = 100, verbose = True):
    '''
    Perform a bootstrap test to compare two Gaussian Mixture models with g0 and g1 components.
    H0: g0 components are sufficient
    H1: g1 components are required (g1 > g0)
    '''

    # fit g0
    model0 = GaussianMixture(n_components=g0, covariance_type=cov_type, max_iter=1000)
    model0.fit(df)
    
    # fit g1
    model1 = GaussianMixture(n_components=g1, covariance_type=cov_type, max_iter=1000)
    model1.fit(df)

    # get LRTS
    ll0 = model0.score(df)
    ll1 = model1.score(df)
    LRTS = 2*(ll1 - ll0)

    # iterate through the samples
    LRTS_list = []
    if verbose: print('Running bootstrap test')
    for s in tqdm.tqdm(range(samples), disable=not verbose):
        # generate bootstrap from model0
        df_s = model0.sample(len(df))[0]

        # fit g0 and g1 to the bootstrap
        model0_s = GaussianMixture(n_components=g0, covariance_type=cov_type, max_iter=1000)
        model0_s.fit(df_s)
        ll0_s = model0_s.score(df_s)
        model1_s = GaussianMixture(n_components=g1, covariance_type=cov_type, max_iter=1000)
        model1_s.fit(df_s)
        ll1_s = model1_s.score(df_s)
        LRTS_s = 2*(ll1_s - ll0_s) # calculate LRTS for this sample
        LRTS_list.append(LRTS_s)

    # calculate p-value
    p_value = np.mean(LRTS > np.array(LRTS_list))

    # save 5th and 95th percentile
    LRTS_list = np.array(LRTS_list)
    percentile_5 = np.percentile(LRTS_list, 5)
    percentile_95 = np.percentile(LRTS_list, 95)

    if verbose:
        print(f'LRTS: {LRTS}, p-value: {p_value}')
        print(f'5th percentile: {percentile_5}, 95th percentile: {percentile_95}')
    
    return p_value
    

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
    # folder_name = 'complete_data/gaussian_mixture_full'
    folder_name = 'resample'
    complete = True
    resample = True

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
    df_errors = pd.read_csv('data/pred_errors.csv', index_col=0)
    trait_labels = df_trait.columns
    species_labels = df_trait.index

    # resample df_trait
    if resample:
        df_trait = np.random.normal(df_trait.values, df_errors.values[:,0])
        # back to dataframe
        df_trait = pd.DataFrame(df_trait, columns=trait_labels, index=species_labels)

    # impute missing values
    if not complete:
        imputer = KNNImputer(n_neighbors=30)
        print('Imputing missing values')
        df_trait = imputer.fit_transform(df_trait)

        # save the imputed data
        data_path = os.path.join(os.getcwd(), 'data')
        df_trait_imputed_save = pd.DataFrame(df_trait, columns=trait_labels, index=species_labels)
        df_trait_imputed_save.to_csv(os.path.join(data_path, 'traits_obs_log_imputed.csv'))

    # n_components_list = [100,200,300,400,500,600]
    # # grid histplot
    # print('Running grid histplot')
    # grid_histplot_clusters(df_trait, n_components_list, output_path, file_name = 'grid_histplot_clusters')
    # exit()

    ### TRIAL ###
    K = 40
    model = GaussianMixture(n_components=K, covariance_type='full', max_iter=1000,
                            n_init=10, random_state=42, verbose=1)
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
    # samples_mixture(df_trait, K, os.path.join(output_path, 'samples'), samples=100)



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
    exit()

    resample = True

    # read data
    folder_name = 'resample'
    output_path = os.path.join(os.getcwd(), 'output', folder_name)

    df_trait = pd.read_csv('data/traits_pred_log.csv', index_col=0)
    df_errors = pd.read_csv('data/pred_errors.csv', index_col=0)
    trait_labels = df_trait.columns
    species_labels = df_trait.index

    # resample df_trait
    if resample:
        # print(df_errors.values[:,0])
        df_trait = np.random.normal(df_trait.values, df_errors.values[:,0])
        # back to dataframe
        df_trait = pd.DataFrame(df_trait, columns=trait_labels, index=species_labels)
        

    # n_component_list = [i for i in range(25,1025, 25)]
    # n_component_list = [1,2,5,10,50] + [i for i in range(100, 2100, 100)]
    # n_component_list = [2,5,10]
    # n_component_list = [10,20,30,40,50,100,150,250,300,350,400,450,
    #                     500,600,700,800,900,1000,1250,1500,2000]
    n_component_list = [i+1 for i in range(20)] + [2*i for i in range(11, 51)]
    print('Components to try')
    print(n_component_list)  
    print('Running Gaussian Mixture')
    run_mixtures(df_trait, output_path, n_component_list, cov_type='full', parallel=True, verbose=True)
   
    # cross_validate_gmm(df_trait, n_component_list, output_path, n_splits=5, cov_type='full', verbose=True)

    # df_trait_sample = df_trait.sample(frac=0.5, random_state=42)


    # # run bootstrap test
    # bootstrap_test(df_trait_sample, 500, 501, samples=20)
    # # print(f'LRTS: {LRTS}, p-value: {p_value}')


