import anndata as ad
import numpy as np
import scanpy as sc
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score


def run_leiden_clustering_on_adata(
    adata_input,
    resolution: float = 1.0,
    key_added: str = "leiden",
    preprocess: bool = True,
):
    """
    Run Leiden clustering on a provided AnnData object.

    Parameters
    ----------
    adata_input : AnnData
        The AnnData object to cluster (can be subset of genes).
    resolution : float
        Resolution parameter for Leiden.
    key_added : str
        Key under which to store clustering result in `.obs`.
    preprocess : bool
        Whether to run normalization, log1p, PCA, and neighbors.

    Returns
    -------
    labels : pd.Series
        The Leiden cluster labels.
    """
    adata = adata_input.copy()

    if preprocess:
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added=key_added,
    )

    return adata.obs[key_added].copy()


def run_leiden_clustering_on_random_gene_subset(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int | None = 100,
    key_prefix: str = "leiden",
    random_state: int = 42,
):
    """
    Run Leiden clustering on either a random subset of genes or all genes.

    Parameters
    ----------
    sdata : SpatialData
        The spatialdata object.
    resolution : float
        Leiden resolution.
    n_genes_subset : int or None
        If int, run on that number of random genes. If None, use all genes.
    key_prefix : str
        Prefix for result key in .obs.
    random_state : int
        Seed for reproducibility (when subsetting genes).

    Returns
    -------
    key_added : str
        The key under which clustering results are stored in .obs.
    """
    adata = sdata.tables["table"]
    key_added = None

    if n_genes_subset is None:
        # Use all genes
        adata_subset = adata
        key_added = f"{key_prefix}_allgenes_res{resolution}"
    else:
        # Use random subset of genes
        rng = np.random.default_rng(random_state)
        n_genes = adata.shape[1]
        if n_genes_subset > n_genes:
            raise ValueError("n_genes_subset cannot be greater than total number of genes")

        gene_indices = rng.choice(n_genes, size=n_genes_subset, replace=False)
        gene_names = adata.var_names[gene_indices]
        adata_subset = adata[:, gene_names]
        key_added = f"{key_prefix}_{n_genes_subset}_res{resolution}_seed{random_state}"

    # Run Leiden and store in original object
    labels = run_leiden_clustering_on_adata(adata_subset, resolution=resolution, key_added=key_added)
    adata.obs[key_added] = labels.values

    return key_added


def compute_pairwise_ari(adata: ad.AnnData, cluster_keys: list[str]) -> float:
    """
    Compute the pairwise adjusted Rand index (ARI) for given cluster keys in an AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing clustering information.
    cluster_keys : List[str]
        The key(s) in `adata.obs` that contain the cluster labels.

    Returns
    -------
    float
        The average pairwise ARI across the specified cluster keys.
    """
    n_clusterings = len(cluster_keys)
    assert n_clusterings > 1, "At least two cluster keys are required to compute pairwise ARI."

    # Ensure all specified cluster keys exist in adata.obs
    for key in cluster_keys:
        if key not in adata.obs:
            raise ValueError(f"Cluster key '{key}' not found in adata.obs.")

    # Compute pairwise ARI scores
    ARI_matrix = np.zeros((n_clusterings, n_clusterings))

    for i in range(n_clusterings):
        for j in range(i + 1, n_clusterings):
            ari = adjusted_rand_score(adata.obs[cluster_keys[i]], adata.obs[cluster_keys[j]])
            ARI_matrix[i, j] = ARI_matrix[j, i] = ari
    np.fill_diagonal(ARI_matrix, 1.0)

    return ARI_matrix


def compute_mean_ari(ari_matrix: np.ndarray) -> float:
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
    return np.mean(upper_triangle)
