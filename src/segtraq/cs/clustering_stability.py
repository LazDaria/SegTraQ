import numpy as np
import pandas as pd
import spatialdata as sd
from sklearn.metrics import silhouette_score

from .utils import (
    _compute_cluster_connectedness,
    compute_mean_ari,
    compute_mean_purity,
    compute_pairwise_ari,
    compute_pairwise_purity,
    run_leiden_clustering_on_random_subset,
)


def compute_cluster_connectedness(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    use_weights: bool = False,
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    inplace: bool = True,
) -> float:
    """
    Compute cluster connectedness for different Leiden clustering resolutions
    and report the best (highest) one.
    If a cell_type_key is provided, compute the connectedness for that clustering only.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float or list of float, optional
        The resolution parameter(s) for Leiden clustering, by default (0.6, 0.8, 1.0).
    use_weights: bool
        Use edge weights to evaluate connectedness. If false, fraction of
        equal neighbors is used.
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
        The best (highest) cluster connectedness across resolutions.
    """
    adata = sdata.tables["table"]

    if isinstance(resolution, float):
        resolution = [resolution]

    best_distance = 0.0
    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )
        labels = adata.obs[cell_type_key].values
        # remove NaN labels
        if len(np.unique(labels[~pd.isna(labels)])) > 1:
            if "connectivities" not in adata.obsp:
                raise ValueError(
                    "Connectivities not found in adata.obsp['connectivities']. "
                    "Please compute neighbors first by running sc.pp.neighbors(adata)."
                )
            distance_val = _compute_cluster_connectedness(adata.obsp["connectivities"], labels)
            return float(distance_val)
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    if "neighbors" not in adata.uns:
        raise ValueError(
            "Neighbors not found in adata. Please compute neighbors first by running sc.pp.neighbors(adata)."
        )

    for res in resolution:
        key_added, _ = run_leiden_clustering_on_random_subset(
            sdata,
            resolution=res,
            frac_cells_subset=1.0,  # Use all cells
            key_prefix=key_prefix,
            random_state=random_state,
            recompute_neighbors=False,
        )
        labels = adata.obs[key_added].values
        if len(np.unique(labels)) > 1:
            distance_val = _compute_cluster_connectedness(adata.obsp["connectivities"], labels, use_weights=use_weights)
            if distance_val > best_distance:
                best_distance = float(distance_val)

    if inplace:
        sdata.tables["table"].uns["cluster_connectedness"] = best_distance

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
    If a cell_type_key is provided, compute the silhouette score for provided labels.

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
        If provided, compute the silhouette score for provided labels.
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

        labels_nn = adata.obs[cell_type_key].dropna()
        if labels_nn.nunique() > 1:  # Ensure more than one cluster exists
            if "X_pca" not in adata.obsm:
                raise ValueError("PCA coordinates not found in adata.obsm['X_pca']. Please run PCA first.")
            # remove NaN labels
            adata_subset = adata[~pd.isna(adata.obs[cell_type_key]), :]
            labels = adata_subset.obs[cell_type_key].values
            silhouette_avg = silhouette_score(adata_subset.obsm["X_pca"], labels, metric=metric)
            best_silhouette_score = float(silhouette_avg)
            key = "silhouette_score_labels"
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    else:
        # ensure that we already have neighbors computed
        # this way we avoid recomputing neighbors multiple times (for the different resolutions)
        if "neighbors" not in adata.uns:
            raise ValueError(
                "Neighbors not found in adata. Please compute neighbors first by running sc.pp.neighbors(adata)."
            )

        key = "silhouette_score"
        for res in resolution:
            # Run clustering for each resolution
            key_added, pca = run_leiden_clustering_on_random_subset(
                sdata,
                resolution=res,
                frac_cells_subset=1.0,  # Use all cells
                key_prefix=key_prefix,
                random_state=random_state,
                recompute_neighbors=False,
            )

            # Compute silhouette score
            labels = adata.obs[key_added]
            labels_nn = labels[~pd.isna(labels)]
            if len(pd.unique(labels_nn)) > 1:  # Ensure more than one cluster exists
                silhouette_avg = silhouette_score(pca, labels, metric=metric)
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg

    if inplace:
        sdata.tables["table"].uns[key] = best_silhouette_score

    return best_silhouette_score


def compute_purity(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    frac_cells_subset: float = 0.63,
    key_prefix: str = "leiden_subset",
    inplace: bool = True,
) -> float:
    """
    Compute the clustering stability using pairwise purity on random subsets of genes.
    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 1.0.
    frac_cells_subset : float, optional
        The fraction of cells to subset for clustering, by default 0.63.
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    inplace : bool, optional
        Whether to store the computed purity in sdata.uns, by default True.

    Returns
    -------
    float
        The average pairwise purity across the specified cluster keys.
    """
    adata = sdata.tables["table"]
    cluster_keys = []

    for random_state in range(5):
        key_added, _pca = run_leiden_clustering_on_random_subset(
            sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
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
    frac_cells_subset: float = 0.63,
    key_prefix: str = "leiden_subset",
    inplace: bool = True,
) -> float:
    """
    Compute the clustering stability using pairwise adjusted Rand index (ARI) on random subset of cells.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 1.0.
    frac_cells_subset : float, optional
        The fraction of cells to subset for clustering, by default 0.63.
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
        key_added, _pca = run_leiden_clustering_on_random_subset(
            sdata,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)
    pairwise_aris = compute_pairwise_ari(adata, cluster_keys)
    mean_ari = float(compute_mean_ari(pairwise_aris))

    if inplace:
        sdata.tables["table"].uns["mean_ari"] = mean_ari

    return mean_ari
