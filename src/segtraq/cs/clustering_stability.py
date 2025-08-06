import spatialdata as sd

from .utils import (
    compute_mean_ari,
    compute_pairwise_ari,
    run_leiden_clustering_on_random_gene_subset,
)


def compute_clustering_stability(
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
