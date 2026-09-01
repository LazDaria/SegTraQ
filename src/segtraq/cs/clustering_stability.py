import warnings

import numpy as np
import pandas as pd
import spatialdata as sd
from sklearn.metrics import silhouette_score as _silhouette_score

from ..constants import CONNECTIVITIES_KEY, NEIGHBORS_KEY, PCA_KEY
from ..utils import _get_pca_and_neighbors, merge_into_uns
from .utils import (
    _cluster_connectedness,
    ari_mean,
    ari_pairwise,
    purity_mean,
    purity_pairwise,
    run_leiden_clustering_on_random_subset,
)


def cluster_connectedness(
    sdata: sd.SpatialData,
    resolution: float | list[float] = 0.2,
    use_weights: bool = False,
    tables_key: str = "table",
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    use_hvg: bool = False,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    target_sum: float | None = None,
    inplace: bool = True,
    leiden_kwargs: dict | None = None,
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
        The resolution parameter(s) for Leiden clustering, by default 0.2.
    use_weights: bool
        Use edge weights to evaluate connectedness. If false, fraction of
        equal neighbors is used.
    tables_key : str, optional
        The key in sdata.tables where the relevant AnnData is stored, by default "table".
    key_prefix : str, optional
        Prefix for clustering keys in .obs, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.
    cell_type_key : str, optional
        If provided, compute cluster connectedness for this clustering only.
    use_hvg: bool, optional
        Whether to use highly variable genes (HVGs) for PCA. By default False.
    n_neighbors: int, optional
        Number of neighbors to use for computing the connectivity matrix. Default is 15.
    n_pcs: int, optional
        Number of principal components to compute for PCA. Default is 50.
    target_sum: float | None, optional
        Target sum for normalization in `scanpy.pp.normalize_total()` before PCA.
        Default is None.
    inplace : bool, optional
        Whether to store the computed cluster connectedness in sdata.uns, by default True.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        For example, `flavor='igraph'` can be used to specify the Leiden implementation.

    Returns
    -------
    float
        The best (highest) cluster connectedness across resolutions.
    """
    adata = sdata.tables[tables_key]

    if isinstance(resolution, float):
        resolution = [resolution]

    best_distance = np.nan
    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )
        labels = adata.obs[cell_type_key].values
        valid_labels = labels[~pd.isna(labels)]
        if len(pd.unique(valid_labels)) > 1:
            if CONNECTIVITIES_KEY not in adata.obsp:
                adata = _get_pca_and_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, target_sum=target_sum)
                sdata.tables[tables_key] = adata
            distance_val = _cluster_connectedness(
                adata.obsp[CONNECTIVITIES_KEY],
                labels,
                use_weights=use_weights,
            )
            return float(distance_val)
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    if NEIGHBORS_KEY not in adata.uns:
        adata = _get_pca_and_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, target_sum=target_sum)
        sdata.tables[tables_key] = adata

    for res in resolution:
        key_added, _, _ = run_leiden_clustering_on_random_subset(
            sdata,
            tables_key=tables_key,
            resolution=res,
            frac_cells_subset=1.0,  # Use all cells
            key_prefix=key_prefix,
            random_state=random_state,
            use_hvg=use_hvg,
            recompute_neighbors=False,
            leiden_kwargs=leiden_kwargs,
        )
        labels = adata.obs[key_added].values
        valid_mask = ~pd.isna(labels)
        valid_labels = labels[valid_mask]

        if len(pd.unique(valid_labels)) > 1:
            # Slice connectivity matrix to valid cells only — both rows AND columns
            connectivity_subset = adata.obsp[CONNECTIVITIES_KEY][np.ix_(valid_mask, valid_mask)]

            distance_val = _cluster_connectedness(
                connectivity_subset,
                valid_labels,
                use_weights=use_weights,
            )
            if np.isnan(best_distance) or distance_val > best_distance:
                best_distance = float(distance_val)
        else:
            warnings.warn(
                f"Leiden clustering at resolution {res} produced only one cluster. Skipping connectedness calculation.",
                stacklevel=2,
            )

    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"cluster_connectedness": best_distance})

    return best_distance


def silhouette_score(
    sdata: sd.SpatialData,
    resolution: float | list[float] = 0.2,
    metric: str = "euclidean",
    tables_key: str = "table",
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
    cell_type_key: str | None = None,
    use_hvg: bool = False,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    target_sum: float | None = None,
    inplace: bool = True,
    leiden_kwargs: dict | None = None,
) -> float:
    """
    Compute the silhouette score for different resolutions and report the best one.
    If a cell_type_key is provided, compute the silhouette score for provided labels.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 0.2.
    metric : str, optional
        The metric to use for silhouette score calculation, by default "euclidean".
    tables_key : str, optional
        The key in sdata.tables where the relevant AnnData is stored, by default "table".
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    random_state : int, optional
        Seed for reproducibility, by default 42.
    cell_type_key : str, optional
        If provided, compute the silhouette score for provided labels.
    use_hvg: bool, optional
        Whether to use highly variable genes (HVGs) for PCA. By default False.
    n_neighbors: int, optional
        Number of neighbors to use for computing the connectivity matrix. Default is 15.
    n_pcs: int, optional
        Number of principal components to compute for PCA. Default is 50.
    target_sum: float | None, optional
        Target sum for normalization in `scanpy.pp.normalize_total()` before PCA.
        Default is None.
    inplace : bool, optional
        Whether to store the computed silhouette score in sdata.uns, by default True.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        For example, `flavor='igraph'` can be used to specify the Leiden implementation.

    Returns
    -------
    float
        The silhouette score of the clustering.
    """
    adata = sdata.tables[tables_key]
    key = None

    best_silhouette_score = np.nan
    if not isinstance(resolution, list | tuple):
        resolution = [resolution]

    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise ValueError(
                f"cell_type_key '{cell_type_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}"
            )

        labels_nn = adata.obs[cell_type_key].dropna()
        if labels_nn.nunique() > 1:  # Ensure more than one cluster exists
            if PCA_KEY not in adata.obsm:
                adata = _get_pca_and_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, target_sum=target_sum)
                sdata.tables[tables_key] = adata
            # remove NaN labels
            adata_subset = adata[~pd.isna(adata.obs[cell_type_key]), :]
            labels = adata_subset.obs[cell_type_key].values
            silhouette_avg = _silhouette_score(adata_subset.obsm[PCA_KEY], labels, metric=metric)
            best_silhouette_score = float(silhouette_avg)
            key = "silhouette_score_labels"

            # handle inplace within the branch and return early, avoiding fall-through
            if inplace:
                merge_into_uns(sdata, tables_key=tables_key, updates={key: best_silhouette_score})
            return best_silhouette_score
        else:
            raise ValueError(f"cell_type_key '{cell_type_key}' must contain more than one cluster")

    else:
        # ensure that we already have neighbors computed
        # this way we avoid recomputing neighbors multiple times (for the different resolutions)
        if NEIGHBORS_KEY not in adata.uns:
            adata = _get_pca_and_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, target_sum=target_sum)
            sdata.tables[tables_key] = adata

        key = "silhouette_score"
        for res in resolution:
            # Run clustering for each resolution
            _, pca, labels = run_leiden_clustering_on_random_subset(
                sdata,
                tables_key=tables_key,
                resolution=res,
                frac_cells_subset=1.0,  # Use all cells
                key_prefix=key_prefix,
                random_state=random_state,
                use_hvg=use_hvg,
                recompute_neighbors=False,
                leiden_kwargs=leiden_kwargs,
            )

            if len(pd.unique(labels)) > 1:  # Ensure more than one cluster exists
                if pca is None:
                    adata = _get_pca_and_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, target_sum=target_sum)
                    sdata.tables[tables_key] = adata
                    pca = adata.obsm[PCA_KEY]

                silhouette_avg = _silhouette_score(pca, labels, metric=metric)

                if np.isnan(best_silhouette_score) or silhouette_avg > best_silhouette_score:
                    best_silhouette_score = float(silhouette_avg)
            else:
                warnings.warn(
                    f"Leiden clustering at resolution {res} produced only one cluster. "
                    f"Skipping silhouette score calculation.",
                    stacklevel=2,
                )

    if inplace and key is not None:
        merge_into_uns(sdata, tables_key=tables_key, updates={key: best_silhouette_score})

    return best_silhouette_score


def purity(
    sdata: sd.SpatialData,
    resolution: float = 0.2,
    frac_cells_subset: float = 0.63,
    tables_key: str = "table",
    key_prefix: str = "leiden_subset",
    use_hvg: bool = False,
    inplace: bool = True,
    leiden_kwargs: dict | None = None,
) -> float:
    """
    Compute the clustering stability using pairwise purity on random subsets of cells.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 0.2.
    tables_key : str, optional
        The key in sdata.tables where the relevant AnnData is stored, by default "table".
    frac_cells_subset : float, optional
        The fraction of cells to subset for clustering, by default 0.63.
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    use_hvg: bool, optional
        Whether to use highly variable genes (HVGs) for PCA. By default False.
    inplace : bool, optional
        Whether to store the computed purity in sdata.uns, by default True.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        For example, `flavor='igraph'` can be used to specify the Leiden implementation.

    Returns
    -------
    float
        The average pairwise purity across the specified cluster keys.
    """
    adata = sdata.tables[tables_key]
    cluster_keys = []

    for random_state in range(5):
        key_added, _, _ = run_leiden_clustering_on_random_subset(
            sdata,
            tables_key=tables_key,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            random_state=random_state,
            leiden_kwargs=leiden_kwargs,
        )
        cluster_keys.append(key_added)

    n_clusters_per_key = [adata.obs[k].nunique() for k in cluster_keys]
    if all(n <= 1 for n in n_clusters_per_key):
        warnings.warn(
            "All clustering results produced only one cluster. Please increase the clustering resolution.",
            stacklevel=2,
        )
        mean_purity = float("nan")
    else:
        purity_matrix = purity_pairwise(adata, cluster_keys)
        mean_purity = float(purity_mean(purity_matrix))

    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"mean_purity": mean_purity})

    return mean_purity


def adjusted_rand_index(
    sdata: sd.SpatialData,
    resolution: float = 0.2,
    frac_cells_subset: float = 0.63,
    tables_key: str = "table",
    key_prefix: str = "leiden_subset",
    use_hvg: bool = False,
    inplace: bool = True,
    leiden_kwargs: dict | None = None,
) -> float:
    """
    Compute the clustering stability using pairwise adjusted Rand index (ARI) on random subset of cells.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing clustering information.
    resolution : float, optional
        The resolution parameter for Leiden clustering, by default 0.2.
    frac_cells_subset : float, optional
        The fraction of cells to subset for clustering, by default 0.63.
    tables_key : str, optional
        The key in sdata.tables where the relevant AnnData is stored, by default "table".
    key_prefix : str, optional
        The prefix for the keys under which the clustering results are stored, by default "leiden_subset".
    use_hvg: bool, optional
        Whether to use highly variable genes (HVGs) for PCA. By default False.
    inplace : bool, optional
        Whether to store the computed ARI in sdata.uns, by default True.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        For example, `flavor='igraph'` can be used to specify the Leiden implementation.

    Returns
    -------
    float
        The average pairwise ARI across the specified cluster keys.
    """
    adata = sdata.tables[tables_key]
    cluster_keys = []

    # Run clustering on random subsets of genes
    for random_state in range(5):
        key_added, _, _ = run_leiden_clustering_on_random_subset(
            sdata,
            tables_key=tables_key,
            resolution=resolution,
            frac_cells_subset=frac_cells_subset,
            key_prefix=key_prefix,
            use_hvg=use_hvg,
            random_state=random_state,
            leiden_kwargs=leiden_kwargs,
        )
        cluster_keys.append(key_added)

    n_clusters_per_key = [adata.obs[k].nunique() for k in cluster_keys]
    if all(n <= 1 for n in n_clusters_per_key):
        warnings.warn(
            "All clustering results produced only one cluster. Please increase the clustering resolution.",
            stacklevel=2,
        )
        mean_ari = float("nan")
    else:
        pairwise_aris = ari_pairwise(adata, cluster_keys)
        mean_ari = float(ari_mean(pairwise_aris))

    if inplace:
        merge_into_uns(sdata, tables_key=tables_key, updates={"mean_ari": mean_ari})

    return mean_ari
