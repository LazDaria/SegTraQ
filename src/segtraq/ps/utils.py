
import spatialdata as sd
import geopandas as gpd
import numpy as np
import pandas as pd
from typing import Literal
from ..rc.region_correlation import compute_cell_nuc_match
from ..utils import filter_cells, _is_background

def _get_filtered_transcripts_df(
    sdata: sd.SpatialData,
    genes: str | list[str] | None,
    cell_type_key: str,
    cell_type_query: str | list[str] | None,
    tables_key: str,
    tables_cell_id_key: str,
    points_key: str,
    points_cell_id_key: str,
    points_gene_key: str,
    points_background_id: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    transcript_df : pd.DataFrame
        Filtered transcript table (computed if dask), background removed.
    tbl : AnnData-like
        `sdata.tables[tables_key]` for convenience (used later for area merge, etc.)
    """
    tbl = sdata.tables[tables_key]

    if cell_type_query is not None:
        query_vals = [cell_type_query] if isinstance(cell_type_query, str) else list(cell_type_query)
        adata = filter_cells(adata=tbl, col=cell_type_key, func=lambda x: x.isin(query_vals))
    else:
        adata = tbl

    cell_ids = adata.obs[tables_cell_id_key]

    pts = sdata.points[points_key]
    pts = pts[pts[points_cell_id_key].isin(cell_ids)]

    if genes is not None:
        if isinstance(genes, str):
            pts = pts[pts[points_gene_key] == genes]
        else:
            pts = pts[pts[points_gene_key].isin(list(genes))]

    transcript_df = pts.compute() if hasattr(pts, "compute") else pts
    if transcript_df.empty:
        raise ValueError(f"No transcripts found after filtering (genes={genes}, cell_type_query={cell_type_query}).")

    is_background = _is_background(transcript_df[points_cell_id_key], points_background_id)
    transcript_df = transcript_df.loc[~is_background]
    if transcript_df.empty:
        raise ValueError("All remaining transcripts were background/unassigned after filtering.")

    return transcript_df, tbl

def _get_cell_geometry_lookup(
    sdata: sd.SpatialData,
    region: Literal["cell", "nucleus"],
    shapes_key: str,
    nucleus_shapes_key: str | None,
    tables_key: str,
    tables_cell_id_key: str,
    points_x_key: str,
    points_y_key: str,
    select_by: Literal["iou", "nucleus_fraction"],
    min_intersection_area: float,
    n_jobs: int,
    use_progress: bool,
    inplace: bool,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Returns
    -------
    centroids_df : pd.DataFrame
        index = cell id, columns = [f"{x}_centroid", f"{y}_centroid"]
    boundary_gdf : GeoDataFrame
        index = cell id, columns = ["geometry"]
    """
    tbl = sdata.tables[tables_key]

    if region == "cell":
        gdf_cells = sdata.shapes[shapes_key][["geometry"]]
        centroids_df = pd.DataFrame(
            {
                f"{points_x_key}_centroid": gdf_cells.geometry.centroid.x.to_numpy(),
                f"{points_y_key}_centroid": gdf_cells.geometry.centroid.y.to_numpy(),
            },
            index=gdf_cells.index,
        )
        return centroids_df, gdf_cells

    gdf_nuc = sdata.shapes[nucleus_shapes_key][["geometry"]]
    shapes_index_name = sdata.shapes[shapes_key].index.name  # cell id index name

    if "best_nuc_id" not in tbl.obs.columns:
        match_df = compute_cell_nuc_match(
            sdata=sdata,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            use_progress=use_progress,
            inplace=inplace,
        )
        id_key = shapes_index_name
    else:
        # use obs cell id column
        match_df = tbl.obs[[tables_cell_id_key, "best_nuc_id"]].copy()
        id_key = tables_cell_id_key

    match_df = match_df[[id_key, "best_nuc_id"]].dropna(subset=["best_nuc_id"]).copy()
    match_df["best_nuc_id"] = match_df["best_nuc_id"].astype(gdf_nuc.index.dtype, copy=False)

    nuc_centroids = pd.DataFrame(
        {
            "best_nuc_id": gdf_nuc.index,
            f"{points_x_key}_centroid": gdf_nuc.geometry.centroid.x.to_numpy(),
            f"{points_y_key}_centroid": gdf_nuc.geometry.centroid.y.to_numpy(),
        }
    )

    centroids_df = match_df.merge(nuc_centroids, on="best_nuc_id", how="left").set_index(id_key)

    boundary_gdf = (
        match_df.merge(gdf_nuc, left_on="best_nuc_id", right_index=True, how="left")
        .set_index(id_key)[["geometry"]]
    )

    return centroids_df, boundary_gdf

def _fisher_pearson_sample_skew(x: np.ndarray) -> float:
    """
    Fisher-Pearson *sample* skewness (common “sample skewness” definition).

    Returns
    -------
    float
        - NaN if n < 3
        - 0.0 if sample variance is 0
        - otherwise:
            g1 = (n / ((n-1)(n-2))) * sum(((xi - mean)/s)^3)
          where s is the sample standard deviation (ddof=1).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.nan
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3))