import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats



def get_paths():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_dir = os.path.join(root_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    figures_dir = os.path.join(root_dir, "output", "figures", "traits")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {
        "trait_file": os.path.join(data_dir, "Estimated_trait_table_with_monos.csv"),
        "tax_file": os.path.join(data_dir, "taxonomic_information.csv"),
        "bgci_old_file": os.path.join(data_dir, "bgci_v1_3_matched_names.csv"),
        "bgci_new_file": os.path.join(data_dir, "global_tree_search_trees_1_7.csv"),
        "traits_pred_log_file": os.path.join(processed_dir, "traits_pred_log.csv"),
        "traits_obs_log_file": os.path.join(processed_dir, "traits_obs_log.csv"),
        "angiosperms_file": os.path.join(processed_dir, "angiosperms.csv"),
        "gymnosperms_file": os.path.join(processed_dir, "gymnosperms.csv"),
        "error_dist_file": os.path.join(processed_dir, "error_pred_dist.pkl"),
        "figures_dir": figures_dir,
    }


def load_data(paths):
    df_trait = pd.read_csv(paths["trait_file"])
    df_tax = pd.read_csv(paths["tax_file"])
    df_bgci_old = pd.read_csv(
        paths["bgci_old_file"],
        usecols=["accepted_bin", "TaxonName"],
        encoding="latin1",
    )
    df_bgci_new = pd.read_csv(
        paths["bgci_new_file"],
        usecols=["TaxonName"],
        encoding="latin1",
    )
    return df_trait, df_tax, df_bgci_old, df_bgci_new


def build_trait_tables(df_trait):
    df_trait_pred = df_trait.pivot_table(
        index="accepted_bin",
        columns="trait",
        values="pred_value",
        aggfunc="mean",
    )
    df_trait_obs = df_trait.pivot_table(
        index="accepted_bin",
        columns="trait",
        values="obs_value",
        aggfunc="mean",
    )
    return df_trait_pred, df_trait_obs


def filter_species(df_trait_pred, df_trait_obs, df_tax, df_bgci_old, df_bgci_new):
    new_species = df_bgci_new["TaxonName"].unique()
    df_bgci_old = df_bgci_old[df_bgci_old["TaxonName"].isin(new_species)]
    new_accepted_bin = df_bgci_old["accepted_bin"].unique()

    non_monocot_list = df_tax.loc[df_tax["mono_fern"] == 0, "accepted_bin"].tolist()

    df_trait_pred = df_trait_pred[df_trait_pred.index.isin(new_accepted_bin)]
    df_trait_pred = df_trait_pred[df_trait_pred.index.isin(non_monocot_list)]

    df_trait_obs = df_trait_obs[df_trait_obs.index.isin(new_accepted_bin)]
    df_trait_obs = df_trait_obs[df_trait_obs.index.isin(non_monocot_list)]

    return df_trait_pred, df_trait_obs

def quick_normality_analysis(df_trait_obs_log, paths=None, save=False):
    """
    Run quick normality diagnostics on log-observed traits.

    Returns:
        pd.DataFrame with test statistics and p-values
    """

    results = []

    for trait in df_trait_obs_log.columns:
        data = df_trait_obs_log[trait].replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < 8:  # too small for reliable tests
            continue

        # Shapiro-Wilk (good default)
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(len(data), 5000)))  
        # (sampling avoids failure on very large n)

        # D’Agostino K² (robust for larger n)
        dagostino_stat, dagostino_p = stats.normaltest(data)

        results.append({
            "trait": trait,
            "n": len(data),
            "shapiro_p": shapiro_p,
            "dagostino_p": dagostino_p,
        })

    df_results = pd.DataFrame(results).set_index("trait")


    if save and paths is not None:
        out_file = os.path.join(paths["figures_dir"], "normality_tests.csv")
        df_results.to_csv(out_file)

    return df_results


def save_log_tables(df_trait_pred, df_trait_obs, paths):
    df_trait_pred_log = np.log(df_trait_pred)
    df_trait_obs_log = np.log(df_trait_obs)

    df_trait_pred_log.to_csv(paths["traits_pred_log_file"])
    df_trait_obs_log.to_csv(paths["traits_obs_log_file"])

    return df_trait_pred_log, df_trait_obs_log


def get_groups(df_tax, species_index, paths):
    angiosperms = df_tax.loc[
        (df_tax["group"] == "Angiosperms") & (df_tax["accepted_bin"].isin(species_index)),
        "accepted_bin",
    ].tolist()

    gymnosperms = df_tax.loc[
        (df_tax["group"] == "Gymnosperms") & (df_tax["accepted_bin"].isin(species_index)),
        "accepted_bin",
    ].tolist()

    pd.DataFrame({"accepted_bin": angiosperms}).to_csv(paths["angiosperms_file"], index=False)
    pd.DataFrame({"accepted_bin": gymnosperms}).to_csv(paths["gymnosperms_file"], index=False)

    return angiosperms, gymnosperms


def clean_error_series(series):
    return series.replace([np.inf, -np.inf], np.nan).dropna().tolist()


def build_error_distribution(df_trait_pred, df_trait_obs, angiosperms, gymnosperms, paths):
    errors_dic = {
        "angiosperm": {},
        "gymnosperm": {},
    }

    full_distribution_errors = []
    full_distribution_angiosperms = []
    full_distribution_gymnosperms = []

    base_idx = df_trait_pred.index.intersection(df_trait_obs.index)
    angio_idx = base_idx.intersection(angiosperms)
    gymno_idx = base_idx.intersection(gymnosperms)

    for trait in df_trait_obs.columns:
        base_errors = np.log(df_trait_pred.loc[base_idx, trait] / df_trait_obs.loc[base_idx, trait])
        base_errors = clean_error_series(base_errors)
        errors_dic[trait] = base_errors
        full_distribution_errors += base_errors

        angio_errors = np.log(df_trait_pred.loc[angio_idx, trait] / df_trait_obs.loc[angio_idx, trait])
        angio_errors = clean_error_series(angio_errors)
        errors_dic["angiosperm"][trait] = angio_errors
        full_distribution_angiosperms += angio_errors

        gymno_errors = np.log(df_trait_pred.loc[gymno_idx, trait] / df_trait_obs.loc[gymno_idx, trait])
        gymno_errors = clean_error_series(gymno_errors)
        errors_dic["gymnosperm"][trait] = gymno_errors
        full_distribution_gymnosperms += gymno_errors

    for trait in df_trait_pred.columns:
        if trait not in df_trait_obs.columns:
            errors_dic[trait] = full_distribution_errors
            errors_dic["angiosperm"][trait] = full_distribution_angiosperms
            errors_dic["gymnosperm"][trait] = full_distribution_gymnosperms

    with open(paths["error_dist_file"], "wb") as f:
        pickle.dump(errors_dic, f)

    return errors_dic


def get_shortened_trait_names():
    return {
        "Leaf density": "Leaf Dens.",
        "Wood density": "Wood Dens.",
        "Root depth": "Root Depth",
        "Specific leaf area": "Spec. Leaf Area",
        "Leaf thickness": "Leaf Thick.",
        "Leaf N per mass": "Leaf N/Mass",
        "Leaf K per mass": "Leaf K/Mass",
        "Leaf P per mass": "Leaf P/Mass",
        "Stem conduit diameter": "Stem Cond. Diam.",
        "Leaf Vcmax per dry mass": "Leaf Vcmax/Mass",
        "Stomatal conductance": "Stomatal Cond.",
        "Leaf area": "Leaf Area",
        "Crown height": "Crown Height",
        "Crown diameter": "Crown Diam.",
        "Tree height": "Tree Height",
        "Seed dry mass": "Seed Mass",
        "Bark thickness": "Bark Thick.",
        "Stem diameter": "Stem Diam.",
    }


def plot_trait_distribution(df_trait_pred, paths, shortened_trait_names):
    linewidth = 6.30045
    plt.rcParams.update({"font.size": 8})
    # plt.rcParams.update({"font.family": "Arial"})

    traits_per_row = 4
    num_rows = int(np.ceil(len(df_trait_pred.columns) / traits_per_row))

    fig, axes = plt.subplots(num_rows, traits_per_row, figsize=(linewidth, linewidth * 1.15))
    axes = np.array(axes).flatten()

    for i, trait in enumerate(df_trait_pred.columns):
        axes[i].hist(np.log(df_trait_pred[trait].dropna()), bins=200, color="green", alpha=0.5)
        axes[i].set_title(shortened_trait_names.get(trait, trait))
        axes[i].tick_params(axis="both", which="both", length=0)
        axes[i].grid(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(paths["figures_dir"], "trait_distribution.pdf"))
    plt.close()


def plot_group_errors(df_trait_pred, df_trait_obs, angiosperms, gymnosperms, paths, shortened_trait_names):
    linewidth = 6.30045
    plt.rcParams.update({"font.size": 8})
    # plt.rcParams.update({"font.family": "Arial"})

    common_traits = list(df_trait_obs.columns)
    angio_idx = df_trait_pred.index.intersection(df_trait_obs.index).intersection(angiosperms)
    gymno_idx = df_trait_pred.index.intersection(df_trait_obs.index).intersection(gymnosperms)

    errors_angiosperms = np.log(df_trait_pred.loc[angio_idx, common_traits]) - np.log(df_trait_obs.loc[angio_idx, common_traits])
    errors_gymnosperms = np.log(df_trait_pred.loc[gymno_idx, common_traits]) - np.log(df_trait_obs.loc[gymno_idx, common_traits])

    errors_angiosperms.columns = [shortened_trait_names.get(col, col) for col in errors_angiosperms.columns]
    errors_gymnosperms.columns = [shortened_trait_names.get(col, col) for col in errors_gymnosperms.columns]

    num_plots = len(errors_angiosperms.columns)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(linewidth * 0.48, linewidth))
    axes = np.array(axes).flatten()

    for i, col in enumerate(errors_angiosperms.columns):
        ax = axes[i]
        errors_angiosperms[col].dropna().hist(bins=100, ax=ax, color="lightcoral")
        ax.set_title(col)
        ax.grid(False)

    if len(axes) > num_plots:
        ax_last = axes[-1]
        ax_last.hist([], color="lightcoral", label="Angiosperms")
        ax_last.legend(loc="upper right")
        ax_last.get_legend().get_frame().set_edgecolor("black")
        ax_last.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(paths["figures_dir"], "angiosperms_error.pdf"))
    plt.close()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(linewidth * 0.48, linewidth))
    axes = np.array(axes).flatten()

    for i, col in enumerate(errors_gymnosperms.columns):
        ax = axes[i]
        errors_gymnosperms[col].dropna().hist(bins=100, ax=ax, color="cornflowerblue")
        ax.set_title(col)
        ax.grid(False)

    if len(axes) > num_plots:
        ax_last = axes[-1]
        ax_last.hist([], color="cornflowerblue", label="Gymnosperms")
        ax_last.legend(loc="upper right")
        ax_last.get_legend().get_frame().set_edgecolor("black")
        ax_last.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(paths["figures_dir"], "gymnosperms_error.pdf"))
    plt.close()


def plot_angiosperm_scatter(df_trait_pred, df_trait_obs, angiosperms, paths):
    angiosperms_in_obs = [a for a in angiosperms if a in df_trait_obs.index and a in df_trait_pred.index]
    errors_angiosperms = np.log(df_trait_pred.loc[angiosperms_in_obs, df_trait_obs.columns]) - np.log(
        df_trait_obs.loc[angiosperms_in_obs, df_trait_obs.columns]
    )

    n_traits = df_trait_obs.shape[1]
    n_cols = 3
    n_rows = math.ceil(n_traits / n_cols)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    axs = np.array(axs).flatten()

    for i, trait in enumerate(df_trait_obs.columns):
        axs[i].scatter(
            np.log(df_trait_obs.loc[angiosperms_in_obs, trait]),
            errors_angiosperms[trait],
            alpha=0.5,
        )
        axs[i].set_title(trait)

        if i // n_cols == n_rows - 1:
            axs[i].set_xlabel("Log Observed Values")

        if i % n_cols == 0:
            axs[i].set_ylabel("Errors (Log Predicted - Log Observed)")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.savefig(os.path.join(paths["figures_dir"], "angiosperms_scatter_errors_vs_obs.pdf"))
    plt.close()


def plot_error_histograms(df_trait_pred, df_trait_obs, paths):
    errors = np.log(df_trait_pred) - np.log(df_trait_obs)
    axes = errors.hist(bins=100, figsize=(20, 20))
    plt.savefig(os.path.join(paths["figures_dir"], "error_histograms_log_difference.pdf"))
    plt.close()

    errors = np.log(df_trait_pred / df_trait_obs)
    axes = errors.hist(bins=100, figsize=(20, 20))
    plt.savefig(os.path.join(paths["figures_dir"], "error_histograms_log_ratio.pdf"))
    plt.close()

    if "Seed dry mass" in df_trait_obs.columns and "Seed dry mass" in df_trait_pred.columns:
        plt.figure()
        (df_trait_obs - df_trait_pred)["Seed dry mass"].dropna().hist(bins=200)
        plt.savefig(os.path.join(paths["figures_dir"], "seed_dry_mass_difference_hist.pdf"))
        plt.close()


def plot_pred_vs_log_pred_distributions(df_trait_pred, df_trait_obs, paths):
    for col in df_trait_obs.columns:
        if col not in df_trait_pred.columns:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist(df_trait_pred[col].dropna(), bins=50)
        axs[1].hist(np.log(df_trait_pred[col].dropna()), bins=50)
        axs[0].set_title("Original")
        axs[1].set_title("Log-transformed")
        fig.suptitle(col)
        safe_name = col.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(paths["figures_dir"], f"distribution_{safe_name}.pdf"))
        plt.close()


def plot_pairplots(df_trait_pred, df_trait_obs, paths):
    some_traits = ["Bark thickness", "Leaf area", "Wood density"]
    some_traits = [col for col in some_traits if col in df_trait_pred.columns and col in df_trait_obs.columns]

    if len(some_traits) < 2:
        return

    pd.plotting.scatter_matrix(df_trait_pred[some_traits].dropna(), figsize=(10, 10), diagonal="hist")
    plt.suptitle("Predicted traits")
    plt.savefig(os.path.join(paths["figures_dir"], "pairplot_predicted_traits.pdf"))
    plt.close()

    pd.plotting.scatter_matrix(df_trait_obs[some_traits].dropna(), figsize=(10, 10), diagonal="hist")
    plt.suptitle("Observed traits")
    plt.savefig(os.path.join(paths["figures_dir"], "pairplot_observed_traits.pdf"))
    plt.close()


def make_all_plots(df_trait_pred, df_trait_obs, df_tax, angiosperms, gymnosperms, paths):
    shortened_trait_names = get_shortened_trait_names()

    plot_trait_distribution(df_trait_pred, paths, shortened_trait_names)
    plot_group_errors(df_trait_pred, df_trait_obs, angiosperms, gymnosperms, paths, shortened_trait_names)
    plot_angiosperm_scatter(df_trait_pred, df_trait_obs, angiosperms, paths)
    plot_error_histograms(df_trait_pred, df_trait_obs, paths)
    plot_pred_vs_log_pred_distributions(df_trait_pred, df_trait_obs, paths)
    plot_pairplots(df_trait_pred, df_trait_obs, paths)

def compute_mae_per_trait(df_trait_pred, df_trait_obs, paths=None, save=False):
    """
    Compute MAE per trait using log(pred) vs log(obs), only on overlapping species.

    Returns:
        pd.Series indexed by trait
    """

    # align on common index and traits
    common_idx = df_trait_pred.index.intersection(df_trait_obs.index)
    common_traits = df_trait_obs.columns.intersection(df_trait_pred.columns)

    # compute log errors
    log_pred = np.log(df_trait_pred.loc[common_idx, common_traits])
    log_obs = np.log(df_trait_obs.loc[common_idx, common_traits])

    abs_errors = (log_pred - log_obs).abs()

    # mean absolute error per trait
    mae = abs_errors.mean(axis=0)

    if save and paths is not None:
        out_file = os.path.join(paths["traits_obs_log_file"].replace("traits_obs_log.csv", "mae_per_trait.csv"))
        mae.to_csv(out_file, header=["mae"])

    return mae

def main(make_plots=True):
    paths = get_paths()

    df_trait, df_tax, df_bgci_old, df_bgci_new = load_data(paths)
    df_trait_pred, df_trait_obs = build_trait_tables(df_trait)

    df_trait_pred, df_trait_obs = filter_species(
        df_trait_pred,
        df_trait_obs,
        df_tax,
        df_bgci_old,
        df_bgci_new,
    )

    save_log_tables(df_trait_pred, df_trait_obs, paths)

    species_index = df_trait_pred.index
    angiosperms, gymnosperms = get_groups(df_tax, species_index, paths)

    build_error_distribution(
        df_trait_pred,
        df_trait_obs,
        angiosperms,
        gymnosperms,
        paths,
    )

    # compute MAE per trait
    mae_per_trait = compute_mae_per_trait(df_trait_pred, df_trait_obs, paths, save=False)
    print(mae_per_trait.round(3))



    df_traits_obs_log = np.log(df_trait_obs)
    normality_results = quick_normality_analysis(df_traits_obs_log, paths, save=False)
    print(normality_results.round(4))

    if make_plots:
        make_all_plots(
            df_trait_pred,
            df_trait_obs,
            df_tax,
            angiosperms,
            gymnosperms,
            paths,
        )


if __name__ == "__main__":
    main(make_plots=True)