import numpy as np
import pandas as pd
import spatialdata as sd
from sklearn.metrics import silhouette_score

from .utils import (
    compute_mean_ari,
    compute_mean_cosine_distance_for_clustering,
    compute_mean_purity,
    compute_pairwise_ari,
    compute_pairwise_purity,
    compute_rmsd_for_clustering,
    run_leiden_clustering_on_random_gene_subset,
)


def compute_rmsd(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    inplace: bool = True,
) -> float:
    """
    Compute RMSD for different Leiden clustering resolutions and report the best (lowest) RMSD.
    If a cell_type_key is provided, compute the RMSD for that clustering only.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float or list of float, optional
        The resolution parameter(s) for Leiden clustering, by default (0.6, 0.8, 1.0).
    key_prefix : str, optional
        Prefix for clustering keys in .obs, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.
    cell_type_key : str, optional
        If provided, compute the RMSD for this clustering only.
    inplace : bool, optional
        Whether to store the computed RMSD in sdata.uns, by default True.

    Returns
    -------
    float
        The best (lowest) RMSD across resolutions.
    """
    adata = sdata.tables["table"]

    if isinstance(resolution, float):
        resolution = [resolution]

    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )
        labels = adata.obs[cell_type_key].values
        # remove NaN labels
        if len(np.unique(labels[~pd.isna(labels)])) > 1:
            if "X_pca" not in adata.obsm:
                raise ValueError("PCA coordinates not found in adata.obsm['X_pca']. Please run PCA first.")
            rmsd_val = compute_rmsd_for_clustering(adata.obsm["X_pca"], labels)
            return float(rmsd_val)
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    if "neighbors" not in adata.uns:
        raise ValueError(
            "Neighbors not found in adata. Please compute neighbors first by running sc.pp.neighbors(adata)."
        )

    best_rmsd = np.inf
    for res in resolution:
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
            recompute_neighbors=False,
        )
        labels = adata.obs[key_added].values
        if len(np.unique(labels)) > 1:
            rmsd_val = compute_rmsd_for_clustering(pca, labels)
            if rmsd_val < best_rmsd:
                best_rmsd = float(rmsd_val)

    best_rmsd = best_rmsd if best_rmsd != np.inf else np.nan

    if inplace:
        sdata.tables["table"].uns["rmsd"] = best_rmsd

    return best_rmsd


def compute_mean_cosine_distance(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    inplace: bool = True,
) -> float:
    """
    Compute mean cosine distance for different Leiden clustering resolutions
    and report the best (lowest) mean cosine distance.
    If a cell_type_key is provided, compute the mean cosine distance for that clustering only.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float or list of float, optional
        The resolution parameter(s) for Leiden clustering, by default (0.6, 0.8, 1.0).
    key_prefix : str, optional
        Prefix for clustering keys in .obs, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.
    cell_type_key : str, optional
        If provided, compute the mean cosine distance for this clustering only.
    inplace : bool, optional
        Whether to store the computed mean cosine distance in sdata.uns, by default True.

    Returns
    -------
    float
        The best (lowest) mean cosine distance across resolutions.
    """
    adata = sdata.tables["table"]

    if isinstance(resolution, float):
        resolution = [resolution]

    best_distance = np.inf
    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )
        labels = adata.obs[cell_type_key].values
        # remove NaN labels
        if len(np.unique(labels[~pd.isna(labels)])) > 1:
            if "X_pca" not in adata.obsm:
                raise ValueError("PCA coordinates not found in adata.obsm['X_pca']. Please run PCA first.")
            distance_val = compute_mean_cosine_distance_for_clustering(adata.obsm["X_pca"], labels)
            return float(distance_val)
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    if "neighbors" not in adata.uns:
        raise ValueError(
            "Neighbors not found in adata. Please compute neighbors first by running sc.pp.neighbors(adata)."
        )

    for res in resolution:
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
            recompute_neighbors=False,
        )
        labels = adata.obs[key_added].values
        if len(np.unique(labels)) > 1:
            distance_val = compute_mean_cosine_distance_for_clustering(pca, labels)
            if distance_val < best_distance:
                best_distance = float(distance_val)

    best_distance = best_distance if best_distance != np.inf else np.nan

    if inplace:
        sdata.tables["table"].uns["mean_cosine_distance"] = best_distance

    return best_distance


def compute_silhouette_score(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    metric: str = "euclidean",
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    inplace: bool = True,
) -> float:
    """
    Compute the silhouette score for different resolutions and report the best one.
    If a cell_type_key is provided, compute the silhouette score for that clustering only.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 1.0.
    metric : str, optional
        The metric to use for silhouette score calculation, by default "euclidean".
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.
    cell_type_key : str, optional
        If provided, compute the silhouette score for this clustering only.
    inplace : bool, optional
        Whether to store the computed silhouette score in sdata.uns, by default True.

    Returns
    -------
    float
        The silhouette score of the clustering.
    """
    adata = sdata.tables["table"]

    best_silhouette_score = -1
    if isinstance(resolution, float):
        resolution = [resolution]

    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )
        labels = adata.obs[cell_type_key]
        if len(set(labels)) > 1:  # Ensure more than one cluster exists
            if "X_pca" not in adata.obsm:
                raise ValueError("PCA coordinates not found in adata.obsm['X_pca']. Please run PCA first.")
            # remove NaN labels
            adata_subset = adata[~pd.isna(adata.obs[cell_type_key]), :]
            labels = adata_subset.obs[cell_type_key].values
            silhouette_avg = silhouette_score(adata_subset.obsm["X_pca"], labels, metric=metric)
            return float(silhouette_avg)
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    # ensure that we already have neighbors computed
    # this way we avoid recomputing neighbors multiple times (for the different resolutions)
    if "neighbors" not in adata.uns:
        raise ValueError(
            "Neighbors not found in adata. Please compute neighbors first by running sc.pp.neighbors(adata)."
        )

    for res in resolution:
        # Run clustering for each resolution
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
            recompute_neighbors=False,
        )

        # Compute silhouette score
        labels = adata.obs[key_added]
        if len(set(labels)) > 1:  # Ensure more than one cluster exists
            silhouette_avg = silhouette_score(pca, labels, metric=metric)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg

    if inplace:
        sdata.tables["table"].uns["silhouette_score"] = best_silhouette_score

    return best_silhouette_score


def compute_purity(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int = 100,
    key_prefix: str = "leiden_subset",
    inplace: bool = True,
) -> float:
    """
    Compute the clustering consistency using pairwise purity scores across
    clustering runs on random gene subsets.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    resolution : float
        Leiden resolution parameter.
    n_genes_subset : int
        Number of genes to use per clustering run.
    key_prefix : str
        Prefix for storing cluster labels in .obs.
    inplace : bool, optional
        Whether to store the computed purity score in sdata.uns, by default True.

    Returns
    -------
    float
        Average pairwise purity score.
    """
    adata = sdata.tables["table"]
    cluster_keys = []

    for random_state in range(5):
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=resolution,
            n_genes_subset=n_genes_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)

    purity_matrix = compute_pairwise_purity(adata, cluster_keys)
    mean_purity = float(compute_mean_purity(purity_matrix))

    if inplace:
        sdata.tables["table"].uns["mean_purity"] = mean_purity

    return mean_purity


def compute_ari(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int = 100,
    key_prefix: str = "leiden_subset",
    inplace: bool = True,
) -> float:
    """
    Compute the clustering stability using pairwise adjusted Rand index (ARI) on random subsets of genes.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 1.0.
    n_genes_subset : int, optional
        The number of genes to subset for clustering, by default 100.
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    inplace : bool, optional
        Whether to store the computed ARI in sdata.uns, by default True.

    Returns
    -------
    float
        The average pairwise ARI across the specified cluster keys.
    """
    adata = sdata.tables["table"]
    cluster_keys = []
    # Run clustering on random subsets of genes
    for random_state in range(5):
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=resolution,
            n_genes_subset=n_genes_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)
    pairwise_aris = compute_pairwise_ari(adata, cluster_keys)
    mean_ari = float(compute_mean_ari(pairwise_aris))

    if inplace:
        sdata.tables["table"].uns["mean_ari"] = mean_ari

    return mean_ari
