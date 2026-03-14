import argparse
import os
import pickle
import numpy as np
import pandas as pd

import hdbscan
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

import torch
from gmm_gpu.gmm import GMM
from sklearn.mixture import GaussianMixture

import tqdm


def fit_hdbscan(X):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusterer.fit(X)
    return clusterer.labels_


def fit_optics(X):
    clusterer = OPTICS(min_samples=5)
    clusterer.fit(X)
    return clusterer.labels_


def create_custom_palette(n_colors):
    palette = (
        sns.color_palette("husl", n_colors=n_colors // 3)
        + sns.color_palette("dark", n_colors=n_colors // 3)
        + sns.color_palette("pastel", n_colors=n_colors // 3)
    )
    return palette


def plot_results(X, labels, output_path, seed, plot_method="PCA"):
    os.makedirs(output_path, exist_ok=True)

    num_classes = len(set(labels))
    palette = create_custom_palette(n_colors=max(num_classes, 3))
    markers = ["o", "s", "D"]

    if plot_method == "PCA":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        x_label = "PCA Component 1"
        y_label = "PCA Component 2"
    elif plot_method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        x_label = "t-SNE Component 1"
        y_label = "t-SNE Component 2"
    elif plot_method == "traits":
        X_reduced = X.copy().values
        x_label = X.columns[0]
        y_label = X.columns[1]
    else:
        raise ValueError(plot_method)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        hue=labels,
        palette=palette,
        legend=None,
        s=10,
        markers=markers,
    )
    plt.title(f"Iteration {seed}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    num_clusters = len(set(labels))
    plt.text(
        0.9,
        0.9,
        f"$G={num_clusters}$",
        horizontalalignment="right",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="grey", alpha=0.5, boxstyle="round,pad=0.5"),
    )
    plt.savefig(os.path.join(output_path, f"{seed}_{plot_method}.png"))
    plt.close()


def make_resample_X(
    df_traits_pred,
    df_traits_obs,
    error_dic,
    gymnosperm,
    angiosperm,
    observed_traits,
    seed,
    error_weight=1.0,
    scale=True,
):
    np.random.seed(seed)

    N_gymnosperm = gymnosperm.shape[0]
    N_angiosperm = angiosperm.shape[0]

    X_s = pd.DataFrame(
        np.ones(df_traits_pred.shape) * np.nan,
        columns=df_traits_pred.columns,
        index=df_traits_pred.index,
    )

    for trait in df_traits_pred.columns:
        sampled_error_gym = np.random.choice(error_dic["gymnosperm"][trait], N_gymnosperm, replace=True) * error_weight
        X_s.loc[gymnosperm, trait] = df_traits_pred.loc[gymnosperm, trait] + sampled_error_gym

        sampled_error_ang = np.random.choice(error_dic["angiosperm"][trait], N_angiosperm, replace=True) * error_weight
        X_s.loc[angiosperm, trait] = df_traits_pred.loc[angiosperm, trait] + sampled_error_ang

        if trait in observed_traits:
            complete_index = np.intersect1d(df_traits_obs[trait].dropna().index, df_traits_pred.index)
            X_s.loc[complete_index, trait] = df_traits_pred.loc[complete_index, trait]

    X_np = X_s.values.astype(np.float32)
    if scale:
        X_np = RobustScaler().fit_transform(X_np).astype(np.float32)

    return X_np


def build_X_batch(
    seeds,
    df_traits_pred,
    df_traits_obs,
    error_dic,
    gymnosperm,
    angiosperm,
    observed_traits,
    error_weight=1.0,
    scale=True,
):
    X_list = []
    for s in seeds:
        X_list.append(
            make_resample_X(
                df_traits_pred=df_traits_pred,
                df_traits_obs=df_traits_obs,
                error_dic=error_dic,
                gymnosperm=gymnosperm,
                angiosperm=angiosperm,
                observed_traits=observed_traits,
                seed=s,
                error_weight=error_weight,
                scale=scale,
            )
        )
    return np.stack(X_list, axis=0)  # (B, N, D)


def find_best_k_per_seed_gpu(X_batch_t, components_list, bic_output_path, device="cuda", max_iter=100, tol=1e-3, reg_covar=1e-6):
    B = X_batch_t.shape[0]
    bic_mat = np.zeros((len(components_list), B), dtype=np.float64)

    for i, k in tqdm.tqdm(enumerate(components_list), total=len(components_list)):
        gmm = GMM(
            n_components=k,
            device=device,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
        )
        gmm.fit(X_batch_t)
        bic_t = gmm.bic(X_batch_t, force_cpu_result=True)
        bic = bic_t.detach().cpu().numpy().reshape(-1)
        bic_mat[i] = bic

        torch.cuda.empty_cache()

    df_bic = pd.DataFrame(bic_mat.T, columns=[f"K_{k}" for k in components_list])
    df_bic.insert(0, "seed_index", np.arange(B))
    df_bic.to_csv(os.path.join(bic_output_path, "BIC_all_seeds.csv"), index=False)

    best_idx = np.argmin(bic_mat, axis=0)
    best_k = np.array([components_list[j] for j in best_idx], dtype=int)
    best_bic = bic_mat[best_idx, np.arange(B)]

    return best_k, best_bic, bic_mat

# this is done in cpu with sklearn as it is better supported for the final predictions
def predict_and_save_labels_per_seed(
    X_batch_t,
    seeds,
    best_k,
    output_path,
    random_assign=True,
    n_assign=10,
    sklearn_seed=0,
):
    seeds = list(seeds)
    best_k = np.asarray(best_k, dtype=int)

    X_batch_np = X_batch_t.detach().cpu().numpy()  # (B, N, D)

    for i, seed in enumerate(seeds):
        k = int(best_k[i])
        print(f"Predicting labels for seed {seed} with K={k}...")
        X_i = X_batch_np[i]  # (N, D)

        gmm = GaussianMixture(n_components=k, random_state=sklearn_seed)
        gmm.fit(X_i)

        if random_assign:
            probs = gmm.predict_proba(X_i)  # (N, K)
            rng = np.random.default_rng(seed)

            for asg in range(n_assign):
                labels = np.array([rng.choice(k, p=probs[n]) for n in range(probs.shape[0])], dtype=int)
                pd.Series(labels).to_csv(
                    os.path.join(output_path, f"labels_seed_{seed}_{asg}.csv"),
                    index=False,
                )
        else:
            labels = gmm.predict(X_i)  # (N,)
            pd.Series(labels).to_csv(
                os.path.join(output_path, f"labels_seed_{seed}_0.csv"),
                index=False,
            )


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    raise ValueError(f"Invalid boolean value: {value}")


def parse_seeds(seeds_str):
    seeds = []

    for part in seeds_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))

    return seeds


def main(seeds, method, error_weight=1.0, plot=False, random_assign=True, scale=True, device="cuda"):
    small_data, subset_size = False, 100
    two_traits = False
    trait_list = ["Wood density", "Leaf area"]
    n_assign = 10

    if method == "hdbscan":
        random_assign = False

    component_list = [i for i in range(10, 61)]

    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, "data")
    method_label = f"{method}_error{error_weight}"
    method_label += "_rnd" * random_assign + "_scl" * scale
    base_output_path = os.path.join(working_dir, "output", "consensus", method_label)

    os.makedirs(base_output_path, exist_ok=True)
    if small_data:
        output_path = os.path.join(base_output_path, f"small_{subset_size}")
    elif two_traits:
        output_path = os.path.join(base_output_path, trait_list[0] + "_" + trait_list[1])
    else:
        output_path = os.path.join(base_output_path, "full_data")
    os.makedirs(output_path, exist_ok=True)

    plot_output_path = os.path.join(output_path, "plots")
    os.makedirs(plot_output_path, exist_ok=True)

    df_traits_pred = pd.read_csv(os.path.join(data_dir, "traits_pred_log.csv"), index_col=0)
    df_traits_obs = pd.read_csv(os.path.join(data_dir, "traits_obs_log.csv"), index_col=0)
    if two_traits:
        df_traits_pred = df_traits_pred[trait_list]
        df_traits_obs = df_traits_obs[trait_list]
    observed_traits = df_traits_obs.columns

    error_dic = pickle.load(open(os.path.join(data_dir, "error_pred_dist.pkl"), "rb"))

    if small_data:
        index_list = df_traits_pred.index[:subset_size]
        df_traits_pred = df_traits_pred.loc[index_list, :]

    gymnosperm = pd.read_csv(os.path.join(data_dir, "gymnosperms.csv"), index_col=0)["accepted_bin"].values
    angiosperm = pd.read_csv(os.path.join(data_dir, "angiosperms.csv"), index_col=0)["accepted_bin"].values
    gymnosperm = np.intersect1d(gymnosperm, df_traits_pred.index)
    angiosperm = np.intersect1d(angiosperm, df_traits_pred.index)

    if method in {"hdbscan", "optics"}:
        for s in seeds:
            X_s = make_resample_X(
                df_traits_pred, df_traits_obs, error_dic,
                gymnosperm, angiosperm, observed_traits,
                seed=s, error_weight=error_weight, scale=scale
            )
            if method == "hdbscan":
                labels = fit_hdbscan(X_s)
            else:
                labels = fit_optics(X_s)
            pd.Series(labels).to_csv(os.path.join(output_path, f"labels_seed_{s}_0.csv"), index=False)
            if plot:
                plot_results(X_s, labels, plot_output_path, s, "PCA")
                plot_results(X_s, labels, plot_output_path, s, "t-SNE")
        return

    X_batch = build_X_batch(
        seeds=seeds,
        df_traits_pred=df_traits_pred,
        df_traits_obs=df_traits_obs,
        error_dic=error_dic,
        gymnosperm=gymnosperm,
        angiosperm=angiosperm,
        observed_traits=observed_traits,
        error_weight=error_weight,
        scale=scale,
    )

    X_batch_t = torch.as_tensor(X_batch, dtype=torch.float32, device=device)

    bic_path = os.path.join(output_path, "BIC")
    os.makedirs(bic_path, exist_ok=True)

    best_k, best_bic, _ = find_best_k_per_seed_gpu(X_batch_t, component_list, bic_path, device=device)
    pd.DataFrame({"seed": seeds, "best_k": best_k, "best_bic": best_bic}).to_csv(
        os.path.join(bic_path, "best_k_per_seed.csv"), index=False
    )

    predict_and_save_labels_per_seed(
        X_batch_t=X_batch_t,
        seeds=seeds,
        best_k=best_k,
        output_path=output_path,
        random_assign=random_assign,
        n_assign=n_assign,
    )

    if plot:
        for i, s in enumerate(seeds):
            X_s = X_batch[i]
            labels_path = os.path.join(output_path, f"labels_seed_{s}_0.csv")
            if os.path.exists(labels_path):
                labels = pd.read_csv(labels_path, header=None).values.reshape(-1)
                plot_results(X_s, labels, plot_output_path, s, "PCA")
                plot_results(X_s, labels, plot_output_path, s, "t-SNE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True, help="1,2,3-6,10")
    parser.add_argument("--method", type=str, default="gmm")
    parser.add_argument("--error_weight", type=float, default=1.0)
    parser.add_argument("--plot", type=str_to_bool, default=False)
    parser.add_argument("--random_assign", type=str_to_bool, default=False)
    parser.add_argument("--scale", type=str_to_bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    main(seeds, args.method, args.error_weight, args.plot, args.random_assign, args.scale, args.device)
