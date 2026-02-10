# import necessary libraries
import numpy as np
import os
import pandas as pd

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# fit HDBSCAN and see how many clusters we get
# import
import hdbscan

# normalization
from sklearn.preprocessing import StandardScaler
# silhouette score
from sklearn.metrics import silhouette_score

# Kmeans
from sklearn.cluster import KMeans
# affinity propagation
from sklearn.cluster import AffinityPropagation


### https://matthew-parker.rbind.io/post/2021-01-16-pytorch-keras-clustering/ ###
def fit_deep_kmeans(X):
    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)

    # Define the autoencoder architecture
    input_layer = keras.layers.Input(shape=(df_scaled.shape[1],))
    encoder = keras.layers.Dense(64, activation='relu')(input_layer)
    encoder = keras.layers.Dense(32, activation='relu')(encoder)
    encoder = keras.layers.Dense(16, activation='relu')(encoder)
    encoded = keras.layers.Dense(8, activation='relu')(encoder)  # Latent space with reduced dimensionality

    decoder = keras.layers.Dense(16, activation='relu')(encoded)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.Dense(64, activation='relu')(decoder)
    decoder = keras.layers.Dense(df_scaled.shape[1], activation='linear')(decoder)

    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(df_scaled, df_scaled, batch_size=32, epochs=50, validation_split=0.2, verbose=0)

    # Extract the encoder model
    encoder_model = keras.models.Model(inputs=input_layer, outputs=encoded)
    encoded_data = encoder_model.predict(df_scaled)

    # Calculate reconstruction error for evaluation (optional)
    reconstructed_data = autoencoder.predict(df_scaled)
    reconstruction_error = np.mean((df_scaled - reconstructed_data)**2, axis=1)
    print("Mean Reconstruction Error:", reconstruction_error.mean())

    # Determine the optimal number of clusters using the elbow method and silhouette score
    n_cluster_list = range(2, 21)
    inertia_list = []
    silhouette_list = []
    
    for n_clusters in n_cluster_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(encoded_data)
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(encoded_data, kmeans.labels_))

    # Choose the optimal number of clusters based on the silhouette score
    n_cluster_optimal_silhouette = n_cluster_list[np.argmax(silhouette_list)]
    print(f'Optimal number of clusters based on silhouette score: {n_cluster_optimal_silhouette}')

    # Fit KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_cluster_optimal_silhouette, random_state=42, n_init=10)
    kmeans.fit(encoded_data)
    labels = kmeans.labels_

    print(f'Number of clusters: {len(np.unique(labels))}')
    
    return labels



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
    output_path = os.path.join(working_dir, 'output', 'consensus', 'deep_Kmeans')
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


    labels = fit_deep_kmeans(X_s)
    
    # Save labels to a CSV file named by the seed
    pd.Series(labels).to_csv(os.path.join(output_path, f'labels_seed_{random_seed}.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HDBSCAN for a specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrapping.')
    args = parser.parse_args()

    main(args.seed)
