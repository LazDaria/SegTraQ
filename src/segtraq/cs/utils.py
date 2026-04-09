import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score, confusion_matrix


def run_leiden_clustering_on_adata(
    adata_input,
    resolution: float = 1.0,
    key_added: str = "leiden",
    use_hvg: bool = False,
    representation: str | None = None,
    recompute_neighbors: bool = True,
    leiden_kwargs: dict | None = None,
):
    """
    Run Leiden clustering on a provided AnnData object. Leiden clustering is performed on the PCA-reduced data.

    Parameters
    ----------
    adata_input : AnnData
        The AnnData object to cluster (can be subset of genes).
    resolution : float
        Resolution parameter for Leiden.
    key_added : str
        Key under which to store clustering result in `.obs`.
    use_hvg: bool, optional
        Whether to use highly variable genes (HVGs) for PCA. By default False.
    representation : str | None, optional
        Key in `adata.obsm` specifying the feature representation used to compute
        the k-nearest neighbor graph before clustering. This is passed to
        `scanpy.pp.neighbors(..., use_rep=representation)`.
        If `None`, a PCA ('X_pca') embedding is computed internally.
    recompute_neighbors : bool
        Whether to recompute neighbors before clustering.
    leiden_kwargs : dict, optional
        Additional keyword arguments to pass to `scanpy.tl.leiden()`.
        For example, `flavor='igraph'` can be used to specify the Leiden implementation.

    Returns
    -------
    labels : pd.Series
        The Leiden cluster labels.
    """
    adata = adata_input.copy()
    if recompute_neighbors:
        if representation is None:
            sc.pp.pca(adata, mask_var="highly_variable" if use_hvg else None)
            sc.pp.neighbors(adata)
        else:
            sc.pp.neighbors(adata, use_rep=representation)

    sc.tl.leiden(
        adata,
        resolution=resolution,
        n_iterations=2,
        key_added=key_added,
        **(leiden_kwargs or {}),
    )

    return adata.obs[key_added].copy(), adata.obsm["X_pca"]


def subset_adata(
    adata: ad.AnnData,
    frac_cells_subset: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    n_cells = adata.shape[0]
    if frac_cells_subset > 1.0:
        raise ValueError("frac_cells_subset must be <= 1.")

    n_cells_subset = int(n_cells * frac_cells_subset)

    cell_idx = rng.choice(n_cells, size=n_cells_subset, replace=False)
    return adata[cell_idx, :], f"cells{n_cells_subset}"


def run_leiden_clustering_on_random_subset(
    sdata: sd.SpatialData,
    tables_key: str,
    resolution: float = 1.0,
    frac_cells_subset: float = 0.63,
    key_prefix: str = "leiden",
    random_state: int = 42,
    use_hvg: bool = False,
    recompute_neighbors: bool = True,
    representation: str | None = None,
    leiden_kwargs: dict | None = None,
):
    adata = sdata.tables[tables_key]

    # --- Perform subsetting --- #
    adata_subset, subset_label = subset_adata(
        adata,
        frac_cells_subset=frac_cells_subset,
        random_state=random_state,
    )

    key_added = f"{key_prefix}_{subset_label}_res{resolution}_seed{random_state}"

    # Run Leiden clustering
    labels, pca = run_leiden_clustering_on_adata(
        adata_subset,
        resolution=resolution,
        key_added=key_added,
        use_hvg=use_hvg,
        recompute_neighbors=recompute_neighbors,
        representation=representation,
        leiden_kwargs=leiden_kwargs,
    )

    # Store labels in the full AnnData
    # For cell subsetting, missing cells get NaN
    full_labels = pd.Series(index=adata.obs_names, dtype=object)
    full_labels.loc[adata_subset.obs_names] = labels.values

    adata.obs[key_added] = full_labels

    return key_added, pca


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
