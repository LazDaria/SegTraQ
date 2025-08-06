import numpy as np
import scanpy as sc
import spatialdata as sd
from sklearn.metrics import silhouette_score

from .utils import (
    compute_mean_ari,
    compute_mean_purity,
    compute_pairwise_ari,
    compute_pairwise_purity,
    compute_rmsd_for_clustering,
    run_leiden_clustering_on_random_gene_subset,
)


def compute_ari(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int = 100,
    key_prefix: str = "leiden_subset",
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

    Returns
    -------
    float
        The average pairwise ARI across the specified cluster keys.
    """
    adata = sdata.tables["table"]
    cluster_keys = []
    # Run clustering on random subsets of genes
    for random_state in range(5):
        key_added = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=resolution,
            n_genes_subset=n_genes_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)
    pairwise_aris = compute_pairwise_ari(adata, cluster_keys)
    mean_ari = compute_mean_ari(pairwise_aris)
    return float(mean_ari)


def compute_silhouette_score(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    metric: str = "euclidean",
    ncomps: int = 30,
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
) -> float:
    """
    Compute the silhouette score for different resolutions and report the best one.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 1.0.
    metric : str, optional
        The metric to use for silhouette score calculation, by default "euclidean".
    ncomps : int, optional
        The number of principal components to use, by default 30.
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    float
        The silhouette score of the clustering.
    """
    adata = sdata.tables["table"]

    best_silhouette_score = -1
    if isinstance(resolution, float):
        resolution = [resolution]

    for res in resolution:
        # Run clustering for each resolution
        key_added = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
        )

        # Compute silhouette score
        labels = adata.obs[key_added]
        if len(set(labels)) > 1:  # Ensure more than one cluster exists
            # check if PCA is available, otherwise compute it using scanpy
            if "X_pca" not in adata.obsm:
                sc.pp.pca(adata, n_comps=ncomps)
            silhouette_avg = silhouette_score(adata.obsm["X_pca"][:, :ncomps], labels, metric=metric)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg

    return best_silhouette_score


def compute_purity(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int = 100,
    key_prefix: str = "leiden_subset",
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

    Returns
    -------
    float
        Average pairwise purity score.
    """
    adata = sdata.tables["table"]
    cluster_keys = []

    for random_state in range(5):
        key_added = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=resolution,
            n_genes_subset=n_genes_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)

    purity_matrix = compute_pairwise_purity(adata, cluster_keys)
    return float(compute_mean_purity(purity_matrix))


def compute_rmsd(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    ncomps: int = 30,
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
) -> float:
    """
    Compute RMSD for different Leiden clustering resolutions and report the best (lowest) RMSD.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float or list of float, optional
        The resolution parameter(s) for Leiden clustering, by default (0.6, 0.8, 1.0).
    ncomps : int, optional
        Number of principal components to use, by default 30.
    key_prefix : str, optional
        Prefix for clustering keys in .obs, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    float
        The best (lowest) RMSD across resolutions.
    """
    adata = sdata.tables["table"]

    if isinstance(resolution, float):
        resolution = [resolution]

    # Compute PCA if missing
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=ncomps)

    best_rmsd = np.inf
    for res in resolution:
        key_added = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
        )
        labels = adata.obs[key_added].values
        if len(np.unique(labels)) > 1:
            rmsd_val = compute_rmsd_for_clustering(adata.obsm["X_pca"][:, :ncomps], labels)
            if rmsd_val < best_rmsd:
                best_rmsd = rmsd_val

    return best_rmsd if best_rmsd != np.inf else np.nan
