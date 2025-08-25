import anndata as ad
import numpy as np
import scanpy as sc
import spatialdata as sd
from sklearn.metrics import adjusted_rand_score, confusion_matrix


def compute_rmsd_for_clustering(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute RMSD (root mean squared deviation) of clusters from their centroids.

    Parameters
    ----------
    embeddings : np.ndarray
        Data matrix (e.g., PCA coordinates), shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample.

    Returns
    -------
    float
        RMSD value (lower means tighter clusters).
    """
    unique_labels = np.unique(labels)
    total_rmsd = 0.0
    total_points = 0

    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        if len(cluster_points) < 2:
            continue  # skip singletons, no spread
        centroid = np.mean(cluster_points, axis=0)
        rmsd = np.sqrt(np.mean(np.sum((cluster_points - centroid) ** 2, axis=1)))
        total_rmsd += rmsd * len(cluster_points)
        total_points += len(cluster_points)

    return total_rmsd / total_points if total_points > 0 else np.nan


def run_leiden_clustering_on_adata(
    adata_input,
    resolution: float = 1.0,
    key_added: str = "leiden",
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

    Returns
    -------
    labels : pd.Series
        The Leiden cluster labels.
    """
    adata = adata_input.copy()
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        key_added=key_added,
    )

    return adata.obs[key_added].copy(), adata.obsm["X_pca"]


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
    labels, pca = run_leiden_clustering_on_adata(adata_subset, resolution=resolution, key_added=key_added)
    adata.obs[key_added] = labels.values

    return key_added, pca


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


def compute_pairwise_purity(adata: ad.AnnData, cluster_keys: list[str]) -> np.ndarray:
    """
    Compute the pairwise purity scores between different clusterings in .obs.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the cluster assignments.
    cluster_keys : List[str]
        List of .obs keys with clustering labels.

    Returns
    -------
    np.ndarray
        Symmetric matrix of pairwise purity scores.
    """
    n = len(cluster_keys)
    purity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            labels_i = adata.obs[cluster_keys[i]]
            labels_j = adata.obs[cluster_keys[j]]
            # Compute purity in both directions and average
            p1 = compute_purity_score(labels_i, labels_j)
            p2 = compute_purity_score(labels_j, labels_i)
            avg_purity = (p1 + p2) / 2
            purity_matrix[i, j] = purity_matrix[j, i] = avg_purity

    np.fill_diagonal(purity_matrix, 1.0)
    return purity_matrix


def compute_mean_purity(purity_matrix: np.ndarray) -> float:
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
    return np.mean(purity_matrix[np.triu_indices(n, k=1)])
