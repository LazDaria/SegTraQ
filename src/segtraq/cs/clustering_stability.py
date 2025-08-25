import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

from .utils import (
    compute_mean_ari,
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
) -> float:
    """
    Compute RMSD for different Leiden clustering resolutions and report the best (lowest) RMSD.

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

    Returns
    -------
    float
        The best (lowest) RMSD across resolutions.
    """
    adata = sdata.tables["table"]

    if isinstance(resolution, float):
        resolution = [resolution]

    best_rmsd = np.inf
    for res in resolution:
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
        )
        labels = adata.obs[key_added].values
        if len(np.unique(labels)) > 1:
            rmsd_val = compute_rmsd_for_clustering(pca, labels)
            if rmsd_val < best_rmsd:
                best_rmsd = float(rmsd_val)

    return best_rmsd if best_rmsd != np.inf else np.nan


def compute_silhouette_score(
    sdata: sd.SpatialData,
    resolution: float | list[float] = (0.6, 0.8, 1.0),
    metric: str = "euclidean",
    ncomps: int = 50,
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
        The number of principal components to use, by default 50.
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
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=res,
            n_genes_subset=None,  # Use all genes
            key_prefix=key_prefix,
            random_state=random_state,
        )

        # Compute silhouette score
        labels = adata.obs[key_added]
        if len(set(labels)) > 1:  # Ensure more than one cluster exists
            silhouette_avg = silhouette_score(pca, labels, metric=metric)
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
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
            sdata,
            resolution=resolution,
            n_genes_subset=n_genes_subset,
            key_prefix=key_prefix,
            random_state=random_state,
        )
        cluster_keys.append(key_added)

    purity_matrix = compute_pairwise_purity(adata, cluster_keys)
    return float(compute_mean_purity(purity_matrix))


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
        key_added, pca = run_leiden_clustering_on_random_gene_subset(
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


# this should probably go into a different module, but for now I will put it here until we have a better structure
def compute_z_plane_correlation(
    sdata: sd.SpatialData,
    quantile: float = 25,
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
    gene_key: str = "feature_name",
) -> pd.DataFrame:
    """
    Compute the Pearson correlation between the top and bottom quantiles of transcripts in the z-plane.

    This function computes the Pearson correlation between the top and bottom quantiles of transcripts
    in the z-plane for each cell. It subsets the transcripts based on the z-coordinate and calculates
    the correlation for each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing transcript data.
    quantile : float, optional
        The quantile to use for bottom and top subsets, by default 25.
    transcript_key : str, optional
        The key for transcripts in sdata.points, by default "transcripts".
    cell_key : str, optional
        The key for cell IDs in sdata.points, by default "cell_id".
    gene_key : str, optional
        The key for gene names in sdata.points, by default "feature_name".

    Returns
    -------
    pd.DataFrame
        A DataFrame with cell IDs as index and Pearson correlations as values.
    """
    z = sdata.points[transcript_key]["z"]

    # Compute percentiles (assuming z is a dask array or similar)
    z_bottom = np.percentile(z.compute(), quantile)
    z_top = np.percentile(z.compute(), 100 - quantile)

    # Subset the original transcripts DataFrame
    transcripts = sdata.points[transcript_key]

    # Bottom subset (z <= quantile percentile)
    bottom_df = transcripts[transcripts["z"] <= z_bottom]

    # Top subset (z >= 1 - quantile percentile)
    top_df = transcripts[transcripts["z"] >= z_top]

    # Force compute if it's a Dask DataFrame
    top_df_pd = top_df.compute() if hasattr(top_df, "compute") else top_df
    bottom_df_pd = bottom_df.compute() if hasattr(bottom_df, "compute") else bottom_df

    top_counts = (
        top_df_pd.groupby([cell_key, gene_key])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index=cell_key, columns=gene_key, values="count")
        .fillna(0)
        .astype(int)
    )

    bottom_counts = (
        bottom_df_pd.groupby([cell_key, gene_key])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index=cell_key, columns=gene_key, values="count")
        .fillna(0)
        .astype(int)
    )

    # Ensure same order of cell_ids and same set of features
    common_cells = top_counts.index.intersection(bottom_counts.index)
    common_features = top_counts.columns.intersection(bottom_counts.columns)

    # Align both dataframes
    top_aligned = top_counts.loc[common_cells, common_features]
    bottom_aligned = bottom_counts.loc[common_cells, common_features]

    # Compute Pearson correlation for each row (cell_id)
    correlations = [pearsonr(top_aligned.loc[cell_id], bottom_aligned.loc[cell_id])[0] for cell_id in common_cells]

    # Create the result dataframe
    correlation_df = pd.DataFrame({"cell_id": common_cells, "correlation": correlations}).set_index("cell_id")

    return correlation_df
