import anndata as ad
import numpy as np
import scanpy as sc
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score


def run_leiden_clustering_on_random_gene_subset(
    sdata: sd.SpatialData,
    resolution: float = 1.0,
    n_genes_subset: int = 100,
    key_prefix: str = "leiden_subset",
    random_state: int = 42,
):
    adata = sdata.tables["table"]
    rng = np.random.default_rng(random_state)

    # --- Subset genes ---
    n_genes = adata.shape[1]
    if n_genes_subset > n_genes:
        raise ValueError("n_genes_subset cannot be greater than total number of genes")

    gene_indices = rng.choice(n_genes, size=n_genes_subset, replace=False)
    gene_names = adata.var_names[gene_indices]

    # --- Subset AnnData object (keep all cells, only subset genes) ---
    adata_subset = adata[:, gene_names].copy()

    # --- Preprocess subset ---
    if "counts" not in adata_subset.layers:
        adata_subset.layers["counts"] = adata_subset.X.copy()
        sc.pp.normalize_total(adata_subset, inplace=True)
        sc.pp.log1p(adata_subset)

    sc.pp.pca(adata_subset)
    sc.pp.neighbors(adata_subset)

    key_added = f"{key_prefix}_{n_genes_subset}_res{resolution}_seed{random_state}"

    sc.tl.leiden(
        adata_subset,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added="leiden",
    )

    # --- Store results back in original AnnData ---
    adata.obs[key_added] = adata_subset.obs["leiden"].values

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
