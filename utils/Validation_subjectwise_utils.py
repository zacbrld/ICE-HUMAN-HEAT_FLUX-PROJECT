from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

from sklearn.cluster import KMeans


def choose_test_subjects_sexwise_kmeans(
    participants_good: pd.DataFrame,
    n_clusters_f: int = 2,
    n_clusters_m: int = 2,
    pc_cols: tuple[str, str] = ("PC1", "PC2"),
    subject_col: str = "Subject",
    sex_col: str = "Sex",
    random_state: int = 0,
    n_init: int = 10,
    prefer_female_subject: str | None = "F7",
    replace_if_selected: str | None = "F11",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Selects a small, representative test set of subjects using KMeans clustering
    in PCA space, performed separately for females and males.

    Strategy:
    - Run KMeans on (PC1, PC2) for each sex independently
    - For each cluster, select the subject closest to the cluster centroid
    - Optionally override the selected female subject to enforce a preferred choice

    Returns a DataFrame containing exactly:
    n_clusters_f female subjects + n_clusters_m male subjects.
    """

    # Unpack PCA column names
    pc1, pc2 = pc_cols

    # Check that all required columns are present
    needed_cols = {subject_col, sex_col, pc1, pc2}
    missing = needed_cols - set(participants_good.columns)
    if missing:
        raise ValueError(f"participants_good is missing columns: {sorted(missing)}")

    # Store selected subjects
    chosen_rows = []

    # Process females and males independently
    for sex_label, n_clusters in [("F", n_clusters_f), ("M", n_clusters_m)]:

        # Subset data for the current sex
        sub_sex = participants_good[participants_good[sex_col] == sex_label].copy()

        # Basic sanity checks
        if sub_sex.empty:
            raise ValueError(f"No rows found for sex '{sex_label}' in participants_good.")
        if len(sub_sex) < n_clusters:
            raise ValueError(
                f"Not enough subjects for sex '{sex_label}' to form {n_clusters} clusters "
                f"(got {len(sub_sex)} subjects)."
            )

        if verbose:
            print(f"\n{sex_label}-group subjects:", sub_sex[subject_col].tolist())

        # Run KMeans in PCA space for this sex
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init
        )

        X_sex = sub_sex[[pc1, pc2]].values
        labels = kmeans.fit_predict(X_sex)

        # Store cluster assignment
        sub_sex["cluster_sex"] = labels

        if verbose:
            print(f"{sex_label}-group clusters:")
            print(sub_sex.groupby("cluster_sex")[subject_col].apply(list))

        # Select one representative subject per cluster
        for cl in range(n_clusters):

            # Subjects belonging to the current cluster
            group = sub_sex[sub_sex["cluster_sex"] == cl].copy()
            center = kmeans.cluster_centers_[cl]

            # Compute Euclidean distance to the cluster centroid
            group["dist_to_center"] = np.linalg.norm(
                group[[pc1, pc2]].values - center,
                axis=1
            )

            # Default choice: subject closest to centroid
            best = group.sort_values("dist_to_center").iloc[0]

            # Optional manual override for female subjects
            if (
                sex_label == "F"
                and prefer_female_subject is not None
                and replace_if_selected is not None
            ):
                in_group = set(group[subject_col].astype(str).tolist())
                if best[subject_col] == replace_if_selected and prefer_female_subject in in_group:
                    best = group[group[subject_col] == prefer_female_subject].iloc[0]

            chosen_rows.append(best)

    # Build final DataFrame of selected test subjects
    chosen_df = pd.DataFrame(chosen_rows)
    chosen_df = chosen_df[
        [subject_col, sex_col, pc1, pc2, "cluster_sex", "dist_to_center"]
    ].copy()

    if verbose:
        print("\nChosen subjects for the test set (2F + 2M):")
        print(chosen_df)

    return chosen_df


def plot_pca_subjects_with_selection(
    participants: pd.DataFrame,
    chosen_df: pd.DataFrame,
    pc_cols: tuple[str, str] = ("PC1", "PC2"),
    subject_col: str = "Subject",
    sex_col: str = "Sex",
    figsize: tuple[int, int] = (8, 7),
    title: str | None = None,
):
    """
    Visualizes participants in PCA space.

    - All subjects are plotted and colored by sex
    - Each subject is labeled
    - Selected test subjects are highlighted with a bold circle
    """

    pc1, pc2 = pc_cols

    plt.figure(figsize=figsize)

    # Color mapping by sex
    color_map = {"F": "tab:pink", "M": "tab:blue"}
    colors = participants[sex_col].map(color_map)

    # Scatter plot of all participants
    plt.scatter(
        participants[pc1],
        participants[pc2],
        c=colors,
        alpha=0.6,
        edgecolor="none",
    )

    # Add text labels for all subjects
    for _, row in participants.iterrows():
        plt.text(
            row[pc1] + 0.02,
            row[pc2] + 0.02,
            row[subject_col],
            fontsize=8,
            color="black",
            alpha=0.8,
        )

    # Highlight selected test subjects
    for _, row in chosen_df.iterrows():
        plt.scatter(
            row[pc1],
            row[pc2],
            s=200,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
        )
        plt.text(
            row[pc1] + 0.02,
            row[pc2] + 0.05,
            row[subject_col],
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Female",
               markerfacecolor="tab:pink", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Male",
               markerfacecolor="tab:blue", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Selected test subject",
               markeredgecolor="black", markerfacecolor="none",
               markersize=9, markeredgewidth=2),
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.xlabel(pc1)
    plt.ylabel(pc2)

    # Default title if none is provided
    if title is None:
        title = "PCA of Participant Characteristics\nAll Subjects and Selected Test Subjects"
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
