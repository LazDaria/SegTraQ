import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import spatialdata as sd
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from ..constants import (
    CONNECTIVITIES_KEY,
    DISTANCES_KEY,
    HVG_KEY,
    NEIGHBORS_KEY,
    NORM_LOG_LAYER,
    PCA_KEY,
)
from ..utils import (
    _compute_hvg_mask,
    _get_norm_log,
    _resolve_use_hvg,
)


def _get_pca_and_neighbors(
    adata: AnnData,
    raw_layer: str | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    target_sum: float | None = 1e4,
    use_hvg: bool | None = None,
    exclude_gene_prefixes: tuple[str, ...] = (),
) -> AnnData:
    """
    Compute (or reuse) PCA and neighbors using the pipeline's norm_log layer.

    All results are stored under namespaced keys so they can be
    distinguished from any externally-computed PCA/neighbors:
    - adata.layers[NORM_LOG_LAYER]
    - adata.var[HVG_KEY] if HVGs are used
    - adata.obsm[PCA_KEY]
    - adata.uns[NEIGHBORS_KEY]
    - adata.obsp[CONNECTIVITIES_KEY], adata.obsp[DISTANCES_KEY]

    Parameters
    ----------
    adata : AnnData
    raw_layer : str or None
        Layer with raw counts. None → use `.X`.
    n_neighbors: int
        Number of neighbors for `sc.pp.neighbors`.
    n_pcs: int
        Number of PCs for `sc.pp.pca` and `sc.pp.neighbors`.
    target_sum: float or None
        If not None, passed as `target_sum` to `sc.pp.normalize_total` when
        computing the norm_log layer. Ignored if the norm_log layer already exists.
    use_hvg : bool or None, default=None
        If `None`, use HVGs automatically when the panel contains more than
        8,000 genes. If `True`, always use HVGs. If `False`, use all genes.
    exclude_gene_prefixes : tuple of str, default=()
        Gene prefixes to exclude from the HVG set. Has no effect if HVGs are
        not used.

    Returns
    -------
    AnnData
        The same object (modified in place), returned for convenience.
    """
    adata = _get_norm_log(
        adata,
        layer=raw_layer,
        target_sum=target_sum,
    )

    resolved_use_hvg = _resolve_use_hvg(
        adata.n_vars,
        use_hvg,
    )

    if resolved_use_hvg and HVG_KEY not in adata.var:
        adata.var[HVG_KEY] = _compute_hvg_mask(adata, exclude_gene_prefixes=exclude_gene_prefixes)

    if PCA_KEY not in adata.obsm:
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            layer=NORM_LOG_LAYER,
            mask_var=HVG_KEY if resolved_use_hvg else None,
            key_added=PCA_KEY,
        )

    if NEIGHBORS_KEY not in adata.uns:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=PCA_KEY,
            key_added=NEIGHBORS_KEY,
        )

    return adata


def _prepare_cs_adata(
    sdata: sd.SpatialData,
    tables_key: str,
    use_hvg: bool | None,
    exclude_gene_prefixes: tuple[str, ...] = (),
    n_neighbors: int = 15,
    n_pcs: int = 50,
    target_sum: float | None = None,
):
    """Prepare non-zero-count cells while reusing stored PCA when possible."""
    adata = _filter_zero_count_cells(sdata.tables[tables_key])
    if adata.n_obs < 2:
        raise ValueError("Fewer than two non-zero-count cells remain for clustering stability analysis.")

    return _get_pca_and_neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        target_sum=target_sum,
        use_hvg=use_hvg,
        exclude_gene_prefixes=exclude_gene_prefixes,
    )


def _validate_resolution(resolution: list[float] | tuple[float, ...] | float | int) -> list[float]:
    if isinstance(resolution, float | int):
        resolution = [resolution]
    for res in resolution:
        if not isinstance(res, float | int):
            raise ValueError(f"Resolution {res} is not a float or int.")
        if res < 0:
            raise ValueError(f"Resolution {res} must be positive.")
    return resolution


def _filter_zero_count_cells(adata: ad.AnnData) -> ad.AnnData:
    """
    Return an AnnData excluding cells with zero total counts.

    If no zero-count cells are present, return the original object so existing
    SegTraQ PCA/neighbors can be reused. If cells are removed, return a copy
    and discard inherited HVG/PCA/neighbor state so it is recomputed on the
    filtered cells.
    """
    total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    mask = total_counts > 0
    if mask.all():
        return adata

    adata_filtered = adata[mask, :].copy()
    adata_filtered.var.drop(columns=[HVG_KEY], errors="ignore", inplace=True)
    adata_filtered.obsm.pop(PCA_KEY, None)
    adata_filtered.uns.pop(NEIGHBORS_KEY, None)
    adata_filtered.obsp.pop(CONNECTIVITIES_KEY, None)
    adata_filtered.obsp.pop(DISTANCES_KEY, None)
    return adata_filtered


def run_leiden_clustering_on_adata(
    adata_input,
    resolution: float = 1.0,
    key_added: str = "leiden",
    recompute_neighbors: bool = True,
    n_neighbors: int = 15,
    leiden_kwargs: dict | None = None,
):
    """
    Run Leiden clustering on a provided AnnData object with a precomputed SegTraQ PCA.

    Parameters
    ----------
    adata_input : AnnData
        The AnnData object to cluster. Can be a subset of cells, but must contain
        precomputed PCA coordinates in `adata_input.obsm[PCA_KEY]`.
    resolution : float
        Resolution parameter for Leiden.
    key_added : str
        Key under which to store clustering results in `.obs`.
    recompute_neighbors : bool
        Whether to recompute the neighbor graph from the precomputed SegTraQ PCA.
        If `False`, the existing SegTraQ neighbor graph is reused.
    n_neighbors : int, default=15
        Number of neighbors used when recomputing the neighbor graph.
        Ignored if `recompute_neighbors=False`.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        By default, `n_iterations=2` is used. This can be overridden via
        `leiden_kwargs`. For example, `flavor="igraph"` can be used to specify
        the Leiden implementation.

    Returns
    -------
    labels : pd.Series
        The Leiden cluster labels.
    """
    adata = adata_input.copy()

    if recompute_neighbors:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=PCA_KEY,
        )
    else:
        adata.uns["neighbors"] = adata.uns[NEIGHBORS_KEY]
        adata.obsp["connectivities"] = adata.obsp[CONNECTIVITIES_KEY]

    # setting the default number of Leiden iterations to 2,
    # while allowing the user to override it via leiden_kwargs
    kwargs = {"n_iterations": 2, **(leiden_kwargs or {})}

    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key_added,
        **kwargs,
    )

    return adata.obs[key_added].copy()


def subset_adata(
    adata: ad.AnnData,
    frac_cells_subset: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    n_cells = adata.shape[0]
    if frac_cells_subset <= 0.0 or frac_cells_subset > 1.0:
        raise ValueError("frac_cells_subset must be in the interval (0, 1].")

    n_cells_subset = int(n_cells * frac_cells_subset)
    if n_cells_subset < 2:
        raise ValueError(
            "frac_cells_subset results in fewer than 2 cells in the subset. "
            "Please increase frac_cells_subset or provide more cells."
        )

    if n_cells_subset == n_cells:
        return adata.copy(), f"cells{n_cells_subset}"

    cell_idx = rng.choice(n_cells, size=n_cells_subset, replace=False)
    return adata[cell_idx, :], f"cells{n_cells_subset}"


def run_leiden_clustering_on_random_subset(
    sdata: sd.SpatialData,
    adata_prepared: ad.AnnData,
    tables_key: str,
    resolution: float = 1.0,
    frac_cells_subset: float = 0.63,
    key_prefix: str = "leiden",
    random_state: int = 42,
    n_neighbors: int = 15,
    leiden_kwargs: dict | None = None,
):
    adata_full = sdata.tables[tables_key]

    adata_subset, subset_label = subset_adata(
        adata_prepared,
        frac_cells_subset=frac_cells_subset,
        random_state=random_state,
    )

    key_added = f"{key_prefix}_{subset_label}_res{resolution}_seed{random_state}"

    labels = run_leiden_clustering_on_adata(
        adata_subset,
        resolution=resolution,
        key_added=key_added,
        recompute_neighbors=frac_cells_subset < 1.0,
        n_neighbors=n_neighbors,
        leiden_kwargs=leiden_kwargs,
    )

    full_labels = pd.Series(
        index=adata_full.obs_names,
        dtype=object,
    )
    full_labels.loc[adata_subset.obs_names] = labels.values
    adata_full.obs[key_added] = full_labels

    return key_added, labels.values


def ari_pairwise(adata: ad.AnnData, cluster_keys: list[str]) -> np.ndarray:
    """
    Compute the pairwise adjusted Rand index (ARI) for given cluster keys in an AnnData object.
    Handles non-overlapping label sets by restricting to rows where both labels exist.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing cluster labels in `.obs`.
    cluster_keys : list of str
        List of keys in `adata.obs` representing different clusterings.
    Returns
    -------
    np.ndarray
        A symmetric matrix of pairwise ARI scores.
    """

    n = len(cluster_keys)
    assert n > 1, "At least two cluster keys are required to compute pairwise ARI."

    # Ensure all keys exist
    for key in cluster_keys:
        if key not in adata.obs:
            raise ValueError(f"Cluster key '{key}' not found in adata.obs.")

    ARI_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            labels_i = adata.obs[cluster_keys[i]]
            labels_j = adata.obs[cluster_keys[j]]

            # Restrict to cells with non-missing labels in both clusterings
            mask = labels_i.notna() & labels_j.notna()

            labels_i_valid = labels_i[mask]
            labels_j_valid = labels_j[mask]

            # If no overlapping labels → ARI undefined → set NaN
            if len(labels_i_valid) == 0:
                ARI_matrix[i, j] = ARI_matrix[j, i] = np.nan
                continue

            ari = adjusted_rand_score(labels_i_valid, labels_j_valid)
            ARI_matrix[i, j] = ARI_matrix[j, i] = ari

    np.fill_diagonal(ARI_matrix, 1.0)
    return ARI_matrix


def ari_mean(ari_matrix: np.ndarray) -> float:
    """
    Compute the mean ARI from the pairwise ARI matrix.

    Parameters
    ----------
    ari_matrix : np.ndarray
        The pairwise ARI matrix.

    Returns
    -------
    float
        The mean ARI value.
    """
    n = ari_matrix.shape[0]
    upper_triangle = ari_matrix[np.triu_indices(n, k=1)]
    return np.nanmean(upper_triangle)


def compute_purity_score(labels_true, labels_pred):
    """
    Compute the purity score between two cluster labelings.

    Parameters
    ----------
    labels_true : array-like
        First clustering labels (can be treated as ground truth).
    labels_pred : array-like
        Second clustering labels (to compare).

    Returns
    -------
    float
        Purity score.
    """
    contingency = confusion_matrix(labels_true, labels_pred)
    return np.sum(np.max(contingency, axis=0)) / np.sum(contingency)


def purity_pairwise(adata: ad.AnnData, cluster_keys: list[str]) -> np.ndarray:
    n = len(cluster_keys)
    purity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            labels_i = adata.obs[cluster_keys[i]]
            labels_j = adata.obs[cluster_keys[j]]

            # Restrict to intersection where both have labels
            mask = labels_i.notna() & labels_j.notna()
            labels_i_valid = labels_i[mask]
            labels_j_valid = labels_j[mask]

            # Handle empty intersections
            if len(labels_i_valid) == 0:
                purity_matrix[i, j] = purity_matrix[j, i] = np.nan
                continue

            p1 = compute_purity_score(labels_i_valid, labels_j_valid)
            p2 = compute_purity_score(labels_j_valid, labels_i_valid)
            purity_matrix[i, j] = purity_matrix[j, i] = (p1 + p2) / 2

    np.fill_diagonal(purity_matrix, 1.0)
    return purity_matrix


def purity_mean(purity_matrix: np.ndarray) -> float:
    """
    Compute the mean of the upper triangle of the purity matrix.

    Parameters
    ----------
    purity_matrix : np.ndarray
        Pairwise purity score matrix.

    Returns
    -------
    float
        Mean pairwise purity score.
    """
    n = purity_matrix.shape[0]
    return np.nanmean(purity_matrix[np.triu_indices(n, k=1)])


def _cluster_connectedness(connectivities: sp.spmatrix, labels: np.ndarray, use_weights: bool = False) -> float:
    """
    Compute how well connected a clustering is in a kNN graph.

    Parameters
    ----------
    connectivities : scipy.sparse.spmatrix
        Sparse connectivity matrix (n_cells x n_cells), e.g. from Scanpy.
        Nonzero entries indicate graph neighbors.
    labels : np.ndarray
        Cluster labels of shape (n_cells,).
    use_weights: bool
        Use edge weights to evaluate connectedness. If false, fraction of
        equal neighbors is used.

    Returns
    -------
    float
        Mean cluster connectedness in [0, 1].
    """

    if not sp.issparse(connectivities):
        raise ValueError("connectivities must be a scipy sparse matrix")

    if connectivities.shape[0] != len(labels):
        raise ValueError("connectivities and labels must have compatible shapes")

    G = connectivities.tocsr()

    labels = np.asarray(labels)
    # Define which cells are labeled (non-missing)
    # to avoid false negatives in comparison below
    labeled_mask = ~pd.isna(labels)

    n = G.shape[0]
    per_cell = np.empty(n)
    per_cell.fill(np.nan)

    for i in range(n):
        if not labeled_mask[i]:
            continue

        start, end = G.indptr[i], G.indptr[i + 1]
        neighbors = G.indices[start:end]

        if len(neighbors) == 0:
            continue

        # Only consider labeled neighbors
        neigh_labeled = labeled_mask[neighbors]
        if not np.any(neigh_labeled):
            continue

        neighbors = neighbors[neigh_labeled]
        same = labels[neighbors] == labels[i]

        if use_weights:
            row_w = G.data[start:end]
            row_w = row_w[neigh_labeled]
            denom = row_w.sum()
            if denom <= 0:
                continue

            per_cell[i] = float(row_w[same].sum() / denom)

        else:
            per_cell[i] = float(np.mean(same))

    return np.nanmean(per_cell)
