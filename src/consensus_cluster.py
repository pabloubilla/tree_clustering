
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from functools import partial
import os 
from multiprocessing import Pool
import pickle

def same_cluster_matrix(labels, n):
    """
    Compute the matrix of same cluster membership
    """
    same_cluster = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                same_cluster[i,j] = 1
    return same_cluster

def fit_gmm(X, n_comp):
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=0)
    gmm.fit(X)
    return gmm  # Return only the model object



if __name__ == '__main__':


    ### TODO: Check why prob matrix is only giving 0s, just changed the sampling approach 
    #### to non-parametric
    ### PARAMETERS ###
    verbose = True
    small_data, subset_size = True, 500
    compute_prob_matrix = True
    prob_assignation = False
    n_bootstrap = 10
    n_component_list = [35,40,45,50,55,60,65,70]
    n_jobs = 8
    # n_component_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # n_component_list = [2]
    
    if verbose: 
        print('Running bootstrap with %d bootstrap samples' % n_bootstrap)
        print('Number of components: %s' % n_component_list)

    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'data')
    output_path = os.path.join(working_dir, 'output')

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



    # define cluster empty matrix to store labels
    cluster_bootstrap = np.zeros((N_obs, n_bootstrap))
    prob_matrix = np.zeros((N_obs, N_obs))
    best_bic_list = []
    # Also we can store one prob matrix for each n_components
    prob_matrix_dic = {n: np.zeros((N_obs, N_obs)) for n in n_component_list}

    # see if bootstrap dir exists in output
    bootstrap_dir = os.path.join(output_path, 'consensus')
    if small_data:
        bootstrap_dir = os.path.join(bootstrap_dir, f'small_{subset_size}')
    if not os.path.exists(bootstrap_dir):
        os.makedirs(bootstrap_dir)


    # Loop over bootstrap
    for s in range(n_bootstrap):

        if verbose:
            print('Bootstrap %d' % s)
        
        # resample traits 
        # X_s = np.random.normal(df_traits.values, df_errors.values[:,0])
        # X_s = np.zeros(df_traits_pred.shape)
        # create a copy of the dataframe but with empty values
        X_s = np.ones(df_traits_pred.shape) * np.nan
        # as a dataframe now
        X_s = pd.DataFrame(X_s, columns=df_traits_pred.columns, index=df_traits_pred.index)

        for index_trait, trait in enumerate(df_traits_pred.columns):
            ## TODO: Change to angiosperm and gymnosperm difference

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
        
            # sampled_error = np.random.choice(error_dic[trait], N_obs, replace=True)
            # print(sampled_error)
            # X_s[:, index_trait] = df_traits_pred[trait].copy() + sampled_error
            # print(X_s[:, index_trait])
            # print(trait)
            # exit()


        partial_fit_gmm = partial(fit_gmm, X_s)

        # GMM_s = {} # dictionary to store the models

        with Pool(n_jobs) as pool:
            models = pool.map(partial_fit_gmm, n_component_list)

        bic_list = [model.bic(X_s) for model in models]
        best_model_ix = np.argmin(bic_list)
        best_n = n_component_list[best_model_ix]
        best_bic = bic_list[best_model_ix]

        # Store the best BIC
        best_bic_list.append([best_n, best_bic])

        # Store the labels
        # labels_s = GMM_s[best_n].predict(X_s)
        labels_s = models[best_model_ix].predict(X_s) # TODO: change to simulation approach
        cluster_bootstrap[:,s] = labels_s

        # Same cluster matrix
        if compute_prob_matrix:
            prob_matrix += same_cluster_matrix(labels_s, N_obs)

            # Store the probability matrix for each n_components
            for n_comp in n_component_list:
                if prob_assignation:
                    ## this is not working correctly ## probabilities all 0
                    prob_cluster = models[n_component_list.index(n_comp)].predict_proba(X_s)
    
                    prob_matrix_s = np.zeros((N_obs, N_obs))
                    for _ in range(10):
                        labels_s = [np.random.choice(n_comp, p=prob_cluster[i,:]) for i in range(N_obs)] # there might be a matricial way to do this
                        prob_matrix_s += same_cluster_matrix(labels_s, N_obs)
                    # print(labels_s)

                else:
                    labels_s = models[n_component_list.index(n_comp)].predict(X_s)
                    prob_matrix_dic[n_comp] += same_cluster_matrix(labels_s, N_obs)
    
    # Save the labels
    df_cluster = pd.DataFrame(cluster_bootstrap, index=df_traits_pred.index).astype(int)
    df_cluster.to_csv(os.path.join(bootstrap_dir, 'cluster_bootstrap.csv'))

    # save best BIC
    df_best_bic = pd.DataFrame(best_bic_list, columns=['n_components', 'bic'])
    df_best_bic.to_csv(os.path.join(bootstrap_dir, 'best_bic.csv'))

    # Compute the probability matrix and save
    if compute_prob_matrix:
        prob_matrix /= n_bootstrap
        df_prob_matrix = pd.DataFrame(prob_matrix, index=df_traits_pred.index, columns=df_traits_pred.index)
        df_prob_matrix.to_csv(os.path.join(bootstrap_dir, 'prob_matrix.csv'))

        # Save the probability matrix for each n_components
        for n_comp in n_component_list:
            prob_matrix_dic[n_comp] /= n_bootstrap
            df_prob_matrix = pd.DataFrame(prob_matrix_dic[n_comp], 
                                          index=df_traits_pred.index, columns=df_traits_pred.index)
            df_prob_matrix.to_csv(os.path.join(bootstrap_dir, f'prob_matrix_{n_comp}.csv'))
            # save as feather
            df_prob_matrix.to_feather(os.path.join(bootstrap_dir, f'prob_matrix_{n_comp}.feather'))





        



