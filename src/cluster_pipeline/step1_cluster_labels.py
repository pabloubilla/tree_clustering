import argparse
import os
import pickle
import numpy as np
import pandas as pd

# clustering
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# plot
import matplotlib.pyplot as plt
import seaborn as sns

# for Parallel computation
from joblib import Parallel, delayed

# Scaler
from sklearn.preprocessing import StandardScaler, RobustScaler

def fit_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusterer.fit(X)
    return clusterer.labels_

def fit_optics(X):
    clusterer = OPTICS(min_samples=5)
    clusterer.fit(X)
    return clusterer.labels_

def fit_gmm(X, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    return gmm.predict(X)

def assign_random_labels(gmm, X):
    probabilities = gmm.predict_proba(X)
    labels = np.array([np.random.choice(gmm.n_components, p=prob) for prob in probabilities])
    return labels

# def find_best_gmm(X, components_list):
#     best_gmm = None
#     best_bic = np.inf
#     best_n_components = 0

#     for n in components_list:
#         gmm = GaussianMixture(n_components=n, random_state=42)
#         gmm.fit(X)
#         bic = gmm.bic(X)
#         if bic < best_bic:
#             best_bic = bic
#             best_gmm = gmm
#             best_n_components = n

#     return best_gmm, best_n_components, best_bic
def find_best_gmm(X, components_list, output, seed, n_jobs=1):
    """
    Find the best Gaussian Mixture Model (GMM) based on BIC score.
    
    Parameters:
    X (ndarray): The input data.
    components_list (list): List of number of components to evaluate.
    n_jobs (int): Number of parallel jobs (default is 1).
    
    Returns:
    best_gmm: The best fitted GMM model.
    best_n_components: The number of components for the best GMM.
    best_bic: The BIC score of the best GMM.
    """
    
    def fit_gmm(n):
        gmm = GaussianMixture(n_components=n, random_state=seed)
        gmm.fit(X)
        bic = gmm.bic(X)
        return gmm, n, bic
    
    if n_jobs == 1:
        # Sequential computation
        results = [fit_gmm(n) for n in components_list]
    else:
        # Parallel computation
        pass
        # this needs fix for running with sh
        # results = Parallel(n_jobs=n_jobs)(delayed(fit_gmm)(n) for n in components_list)
    bic_list = [r[2] for r in results]
    df_bic = pd.DataFrame(np.array([components_list, bic_list]).T, columns = ['N_components', 'BIC'])
    df_bic.to_csv(os.path.join(output,f'BIC_{seed}.csv'))

    # Find the best result based on BIC
    best_gmm, best_n_components, best_bic = min(results, key=lambda x: x[2])
    
    return best_gmm, best_n_components, best_bic



def find_best_gmm_iterative(X, start_components):
    def compute_bic(n):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        return gmm.bic(X)

    best_bic = compute_bic(start_components)
    best_n_components = start_components

    # Search upwards
    n = start_components + 1
    while True:
        bic = compute_bic(n)
        if bic < best_bic:
            best_bic = bic
            best_n_components = n
            n += 1
        else:
            break

    # Search downwards
    n = start_components - 1
    while n > 0:
        bic = compute_bic(n)
        if bic < best_bic:
            best_bic = bic
            best_n_components = n
            n -= 1
        else:
            break

    best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
    best_gmm.fit(X)
    
    return best_gmm, best_n_components, best_bic

def create_custom_palette(n_colors):
    # Create a custom palette with a mix of pastel, dark, and neutral colors
    palette = sns.color_palette("husl", n_colors=n_colors//3) + \
              sns.color_palette("dark", n_colors=n_colors//3) + \
              sns.color_palette("pastel", n_colors=n_colors//3)
    return palette

def plot_results(X, labels, output_path, seed, plot_method='PCA'):
    os.makedirs(output_path, exist_ok=True)
    
    # Define a categorical palette with many different colors
    num_classes = len(set(labels))
    # if plot_method == 'traits': ## This should probably be adjusted in a way to consider small aggregations
    #     palette = sns.color_palette("husl", num_classes)  # Adjusting to use only the number of unique classes
    # else:
    palette = create_custom_palette(n_colors=num_classes)  # Using a custom palette for PCA and t-SNE
    # palette = create_custom_palette(n_colors=3)

    # Define markers for variety in the scatter plot
    markers = ['o', 's', 'D']

    if plot_method == 'PCA':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        x_label = 'PCA Component 1'
        y_label = 'PCA Component 2'
    elif plot_method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        x_label = 't-SNE Component 1'
        y_label = 't-SNE Component 2'
    elif plot_method == 'traits':
        X_reduced = X.copy().values
        x_label = X.columns[0]
        y_label = X.columns[1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette=palette, legend=None, s=10,
                    markers=markers)
    plt.title(f'Iteration {seed}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add a text box with the number of clusters
    num_clusters = len(set(labels))
    plt.text(0.9, 0.9, f'$G={num_clusters}$', horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.savefig(os.path.join(output_path, f'{seed}_{plot_method}.png'))
    plt.close()


def main(random_seed, method, error_weight = 1.0, n_components=5, plot=True, random_assign = True, scale = True):
    np.random.seed(random_seed)
    verbose = True
    small_data, subset_size = True, 100
    two_traits = False
    trait_list = ['Wood density', 'Leaf area']
    n_assign = 10
    if method == 'hdbscan': random_assign = False # Hdbscan is always deterministic
    # component_list = [10 + 2*i for i in range(30)]
    component_list = [i for i in range(10,61)]
    # component_list = [20,25,30,35,40,45,50,55,60,65,70]
    # component_list = [2,4,6]

    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'data')
    method_label = f'{method}_error{error_weight}'
    method_label += '_rnd'*random_assign + '_scl'*scale
    print(method_label, random_assign)
    base_output_path = os.path.join(working_dir, 'output', 'consensus', method_label)

    os.makedirs(base_output_path, exist_ok=True)
    if small_data:
        output_path = os.path.join(base_output_path, f'small_{subset_size}')
    elif two_traits:
        output_path = os.path.join(base_output_path, trait_list[0] + '_' +  trait_list[1])
    else:
        output_path = os.path.join(base_output_path, 'full_data')
    os.makedirs(output_path, exist_ok=True)

    # Create plotting subfolder
    plot_output_path = os.path.join(output_path, 'plots')
    os.makedirs(plot_output_path, exist_ok=True)

    # Load the data
    df_traits_pred = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0)
    df_traits_obs = pd.read_csv(os.path.join(data_dir, 'traits_obs_log.csv'), index_col=0)
    if two_traits:
        df_traits_pred = df_traits_pred[trait_list]
        df_traits_obs = df_traits_obs[trait_list]
    observed_traits = df_traits_obs.columns

    error_dic = pickle.load(open(os.path.join(data_dir, 'error_pred_dist.pkl'), 'rb'))

    if small_data:
        index_list = df_traits_pred.index[:subset_size]
        df_traits_pred = df_traits_pred.loc[index_list,:]
    N_obs = df_traits_pred.shape[0]

    gymnosperm = pd.read_csv(os.path.join(data_dir, 'gymnosperms.csv'), index_col=0)['accepted_bin'].values
    angiosperm = pd.read_csv(os.path.join(data_dir, 'angiosperms.csv'), index_col=0)['accepted_bin'].values
    gymnosperm = np.intersect1d(gymnosperm, df_traits_pred.index)
    angiosperm = np.intersect1d(angiosperm, df_traits_pred.index)
    N_gymnosperm = gymnosperm.shape[0]
    N_angiosperm = angiosperm.shape[0]

    X_s = pd.DataFrame(np.ones(df_traits_pred.shape) * np.nan, columns=df_traits_pred.columns, index=df_traits_pred.index)

    for trait in df_traits_pred.columns:
        sampled_error_gym = np.random.choice(error_dic['gymnosperm'][trait], N_gymnosperm, replace=True) * error_weight
        X_s.loc[gymnosperm, trait] = df_traits_pred.loc[gymnosperm, trait] + sampled_error_gym

        sampled_error_ang = np.random.choice(error_dic['angiosperm'][trait], N_angiosperm, replace=True) * error_weight
        X_s.loc[angiosperm, trait] = df_traits_pred.loc[angiosperm, trait] + sampled_error_ang

        if trait in observed_traits:
            complete_index = np.intersect1d(df_traits_obs[trait].dropna().index, df_traits_pred.index)
            X_s.loc[complete_index, trait] = df_traits_pred.loc[complete_index, trait]

    # SCALING

    if scale:
        if random_seed == 1:
            print('Scaling variables')
        X_s = RobustScaler().fit_transform(X_s)


    if method == 'hdbscan':
        labels = fit_hdbscan(X_s)
        # Save labels to a CSV file named by the seed
        pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}_0.csv'), index=False)
    if method == 'optics':
        labels = fit_optics(X_s)
        # Save labels to a CSV file named by the seed
        pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}_0.csv'), index=False)
    elif method == 'gmm':
        bic_path = os.path.join(output_path, 'BIC')
        os.makedirs(bic_path, exist_ok = True)
        gmm, n_components, bic = find_best_gmm(X_s, component_list, bic_path, random_seed)
        print('N components: ', n_components)
        if random_assign:
            for asg in range(n_assign):
                labels = assign_random_labels(gmm, X_s)
                # Save labels to a CSV file named by the seed
                pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}_{asg}.csv'), index=False)

        else:
            labels = gmm.predict(X_s)
            # Save labels to a CSV file named by the seed
            pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}_0.csv'), index=False)

    # Plotting
    if plot:
        
        plot_results(X_s, labels, plot_output_path, random_seed, 'PCA')
        plot_results(X_s, labels, plot_output_path, random_seed, 't-SNE')
        if two_traits:
            plot_results(X_s, labels, plot_output_path, random_seed, 'traits')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', '0', 'no', 'off'}:
        return False
    if value.lower() in {'true', '1', 'yes', 'on'}:
        return True
    raise ValueError(f'Invalid boolean value: {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clustering with a specified seed and method.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrapping.')
    parser.add_argument('--method', type=str, default='gmm', help='Clustering method to use.')
    parser.add_argument('--error_weight', type=float, default=1.0, help='Error weight for sampling.')
    parser.add_argument('--components', type=int, default=30, help='Number of components for GMM.')
    parser.add_argument('--plot', type=str_to_bool, default=False, help='Plot the results using PCA and t-SNE.')
    parser.add_argument('--random_assign', type = str_to_bool, default = True)
    parser.add_argument('--scale', type = str_to_bool, default = True)
    args = parser.parse_args()

    # random_seed, method, error_weight = 1.0, n_components=5, plot=True, random_assign = True, scale = True
    main(args.seed, args.method, args.error_weight, args.components, args.plot, args.random_assign, args.scale)





# import argparse
# import numpy as np
# import pandas as pd
# import hdbscan
# from sklearn.mixture import GaussianMixture
# import os
# import pickle

# def fit_hdbscan(X):
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
#     clusterer.fit(X)
#     return clusterer.labels_

# def fit_gmm(X, n_components):
#     gmm = GaussianMixture(n_components=n_components, random_state=42)
#     gmm.fit(X)
#     return gmm.predict(X)

# def find_best_gmm(X, components_list):
#     best_gmm = None
#     best_bic = np.inf
#     best_n_components = 0

#     for n in components_list:
#         gmm = GaussianMixture(n_components=n, random_state=42)
#         gmm.fit(X)
#         bic = gmm.bic(X)
#         if bic < best_bic:
#             best_bic = bic
#             best_gmm = gmm
#             best_n_components = n

#     return best_gmm, best_n_components, best_bic


# def find_best_gmm_iterative(X, start_components):
#     ### Look iteratively
#     def compute_bic(n):
#         gmm = GaussianMixture(n_components=n, random_state=42)
#         gmm.fit(X)
#         return gmm.bic(X)

#     best_bic = compute_bic(start_components)
#     best_n_components = start_components

#     # Search upwards
#     n = start_components + 1
#     while True:
#         bic = compute_bic(n)
#         if bic < best_bic:
#             best_bic = bic
#             best_n_components = n
#             n += 1
#         else:
#             break

#     # Search downwards
#     n = start_components - 1
#     while n > 0:
#         bic = compute_bic(n)
#         if bic < best_bic:
#             best_bic = bic
#             best_n_components = n
#             n -= 1
#         else:
#             break

#     best_gmm = GaussianMixture(n_components=best_n_components, random_state=42)
#     best_gmm.fit(X)
    
#     return best_gmm, best_n_components, best_bic



# def main(random_seed, method, error_weight, n_components=5):
#     np.random.seed(random_seed)
#     verbose = True
#     small_data, subset_size = False, 120
#     # component_list = [10,20,30,40,50,60]
#     component_list = [28 + 2*i for i in range(10)]

#     working_dir = os.getcwd()
#     data_dir = os.path.join(working_dir, 'data')
#     # method label with error weight
#     method_label = f'{method}_error{error_weight}'
#     # if error_weight < 1:
#     #     method_label = f'{method}_error{error_weight}'
#     # else: 
#     #     method_label = method
#     base_output_path = os.path.join(working_dir, 'output', 'consensus', method_label)

#     # Create method specific output path
#     # if not os.path.exists(base_output_path):
#     os.makedirs(base_output_path, exist_ok=True)
#     if small_data:
#         output_path = os.path.join(base_output_path, f'small_{subset_size}')
#     else:
#         output_path = os.path.join(base_output_path, 'full_data')
#     # if not os.path.exists(output_path):
#     os.makedirs(output_path, exist_ok=True)

#     # Load the data
#     df_traits_pred = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0)
#     df_traits_obs = pd.read_csv(os.path.join(data_dir, 'traits_obs_log.csv'), index_col=0)
#     observed_traits = df_traits_obs.columns

#     ### Load the error distribution ###
#     error_dic = pickle.load(open(os.path.join(data_dir, 'error_pred_dist.pkl'), 'rb'))

#     if small_data:
#         index_list = df_traits_pred.index[:subset_size]
#         df_traits_pred = df_traits_pred.loc[index_list,:]
#     N_obs = df_traits_pred.shape[0]

#     ### Load the gymnosperm and angiosperm data ###
#     gymnosperm = pd.read_csv(os.path.join(data_dir, 'gymnosperms.csv'), index_col=0)['accepted_bin'].values
#     angiosperm = pd.read_csv(os.path.join(data_dir, 'angiosperms.csv'), index_col=0)['accepted_bin'].values
#     gymnosperm = np.intersect1d(gymnosperm, df_traits_pred.index)
#     angiosperm = np.intersect1d(angiosperm, df_traits_pred.index)
#     N_gymnosperm = gymnosperm.shape[0]
#     N_angiosperm = angiosperm.shape[0]

#     X_s = pd.DataFrame(np.ones(df_traits_pred.shape) * np.nan, columns=df_traits_pred.columns, index=df_traits_pred.index)

#     for trait in df_traits_pred.columns:
#         sampled_error_gym = np.random.choice(error_dic['gymnosperm'][trait], N_gymnosperm, replace=True)*error_weight
#         X_s.loc[gymnosperm, trait] = df_traits_pred.loc[gymnosperm, trait] + sampled_error_gym

#         sampled_error_ang = np.random.choice(error_dic['angiosperm'][trait], N_angiosperm, replace=True)*error_weight
#         X_s.loc[angiosperm, trait] = df_traits_pred.loc[angiosperm, trait] + sampled_error_ang

#         if trait in observed_traits:
#             complete_index = np.intersect1d(df_traits_obs[trait].dropna().index, df_traits_pred.index)
#             X_s.loc[complete_index, trait] = df_traits_pred.loc[complete_index, trait]

#     if method == 'hdbscan':
#         labels = fit_hdbscan(X_s)
#     elif method == 'gmm':
#         # labels = fit_gmm(X_s, n_components)
#         gmm, n_components, bic = find_best_gmm(X_s, component_list)
#         # gmm2, n_components2, bic2 = find_best_gmm_iterative(X_s, 35)
#         labels = gmm.predict(X_s)
#         print('N components: ', n_components)
#         # print('N iterateive: ', n_components2)

#     # Save labels to a CSV file named by the seed
#     pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}.csv'), index=False)

# ### check this code runs right in shell, check if both methods need different ways of computing the final consensus matrix
# ### keep the number of samples somehow
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run clustering with a specified seed and method.')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrapping.')
#     parser.add_argument('--method', type=str, default='hdbscan', choices=['hdbscan', 'gmm'], help='Clustering method to use.')
#     parser.add_argument('--error_weight', type=float, default=1, help='Clustering method to use.')
#     parser.add_argument('--components', type=int, default=30, help='Number of components for GMM.')
#     args = parser.parse_args()

#     main(args.seed, args.method, args.error_weight, args.components)
