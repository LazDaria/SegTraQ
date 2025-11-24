import numpy as np
import pandas as pd
import spatialdata as sd
from pandas import Series
from geopandas import GeoDataFrame, GeoSeries
from scipy.stats import pearsonr
from rtree.index import Index
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.validation import make_valid

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
            tables_cell_id_key=points_cell_id_key,
            df_cell_id_key=points_cell_id_key,
            fillna_cols=["correlation"],
        )

    return correlation_df


#code adapt from Daria Lazic
def _process_cell(
    cell_row: Series,
    shapes_cell_id_key: str | None,
    id_key: str | None,
    cell_boundaries: GeoDataFrame,
    cell_sindex: Index,
) -> list:
    """For one cell polygon compute the IoU with overlapping cells in z"""
    
    cell_id = cell_row[shapes_cell_id_key] if shapes_cell_id_key is not None else cell_row.name
    
    cell_geom1 = make_valid(cell_row.geometry)
    # Get candidate cell bounding boxes that overlap this cell's bbox
    candidate_idx = list(cell_sindex.intersection(cell_geom1.bounds))
    
    # if there are no candidates, return 0.0
    if not candidate_idx:
        return {"cell_id": cell_id, "IoU": 0.0, "IoU_sum": 0.0}

    candidates = cell_boundaries.iloc[candidate_idx]
    # go over candidates per cell and calculate IoU
    IoU_ls = []
    for _, cell in candidates.iterrows():
        cell_id2 = cell[shapes_cell_id_key] if shapes_cell_id_key is not None else cell.name
        #if it is the same cell, break
        if cell_id == cell_id2:
            break
        cell_geom2 = make_valid(cell.geometry)
        intersection = cell_geom1.intersection(cell_geom2).area
        union = cell_geom1.union(cell_geom2).area
        IoU = intersection / union if union > 0 else 0.0
        IoU_ls.append(IoU)    
    IoU_ls.sort(reverse=True)

    return {"cell_id": cell_id, "IoU": IoU_ls, "IoU_sum": sum(IoU_ls)}

#code adapt from Daria Lazic
def compute_cell_cell_IoU(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    n_jobs: int = -1,
    use_progress: bool = True,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell IoU between cell and nearby cell boundaries in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table. Gene names in
        `sdata.tables[tables_key].var.index` should match the gene field in
        `sdata.points[points_key]` (see `points_gene_key`).
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    shapes_cell_id_key : str,  default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    n_jobs : int, optional
        Number of parallel jobs. Default=-1 uses all CPUs.
    use_progress : bool, optional
        Whether to display a progress bar with tqdm.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
    """

    # Get GeoDataFrames
    cell_boundaries = sdata.shapes[shapes_key]
    # Build spatial index once
    cell_sindex = cell_boundaries.sindex
    # Iterator for cells
    iterator = cell_boundaries.iterrows()
    if use_progress:
        iterator = tqdm(
            iterator,
            total=len(cell_boundaries),
            desc="Processing IoU between overlapping cells",
        )

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
    elif cell_boundaries.index.name is not None:
        id_key = cell_boundaries.index.name
    else:
        id_key = tables_cell_id_key

    # Parallel loop over cells
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(_process_cell)(
            cell_row=cell_row,
            shapes_cell_id_key=shapes_cell_id_key,
            id_key=id_key,
            cell_boundaries=cell_boundaries,
            cell_sindex=cell_sindex,
        )
        for _, cell_row in iterator
    ) 
    IoU_df = pd.DataFrame(results).set_index(
        tables_cell_id_key
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=IoU_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return IoU_df