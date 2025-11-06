import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.stats import pearsonr

from ..utils import merge_into_obs


def compute_z_plane_correlation(
    sdata: sd.SpatialData,
    quantile: float = 25,
    points_key: str = "transcripts",
    points_z_key: str = "z",
    tables_key: str = "table",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    inplace: bool = True,
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
    points_key : str, optional
        The key for transcripts in sdata.points, by default "transcripts".
    points_z_key : str, optional
        The key for z-coordinates in sdata.points, by default "z".
    tables_key : str, optional
        The key for tables in sdata.tables, by default "table".
    points_cell_id_key : str, optional
        The key for cell IDs in sdata.points, by default "cell_id".
    points_gene_key : str, optional
        The key for gene names in sdata.points, by default "feature_name".
    inplace : bool, optional
        Whether to store the computed correlations in sdata.uns, by default True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with cell IDs as index and Pearson correlations as values.
    """
    z = sdata.points[points_key][points_z_key]

    # Compute percentiles (assuming z is a dask array or similar)
    z_bottom = np.percentile(z.compute(), quantile)
    z_top = np.percentile(z.compute(), 100 - quantile)

    # Subset the original transcripts DataFrame
    transcripts = sdata.points[points_key]

    # Bottom subset (z <= quantile percentile)
    bottom_df = transcripts[transcripts[points_z_key] <= z_bottom]

    # Top subset (z >= 1 - quantile percentile)
    top_df = transcripts[transcripts[points_z_key] >= z_top]

    # Force compute if it's a Dask DataFrame
    top_df_pd = top_df.compute() if hasattr(top_df, "compute") else top_df
    bottom_df_pd = bottom_df.compute() if hasattr(bottom_df, "compute") else bottom_df

    top_counts = (
        top_df_pd.groupby([points_cell_id_key, points_gene_key], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .pivot(index=points_cell_id_key, columns=points_gene_key, values="count")
        .fillna(0)
        .astype(int)
    )

    bottom_counts = (
        bottom_df_pd.groupby([points_cell_id_key, points_gene_key], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .pivot(index=points_cell_id_key, columns=points_gene_key, values="count")
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
    correlation_df = pd.DataFrame({points_cell_id_key: common_cells, "correlation": correlations}).set_index(
        points_cell_id_key
    )

    if inplace:
        merge_into_obs(
            sdata,
            tables_key=tables_key,
            df_to_merge=correlation_df,
            table_cell_id_key=points_cell_id_key,
            df_cell_id_key=points_cell_id_key,
            fillna_cols=["correlation"],
        )

    return correlation_df
