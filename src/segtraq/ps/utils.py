from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd

from ..rd.region_difference import match_nuclei_to_cells


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
    parallel_backend: str,
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

    if "nucleus_id" not in tbl.obs.columns:
        match_df = match_nuclei_to_cells(
            sdata=sdata,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            inplace=inplace,
        )

        # re-fetching the table after in-place modification
        tbl = sdata.tables[tables_key]

    match_df = tbl.obs[[tables_cell_id_key, "nucleus_id"]].copy()

    match_df = match_df[[tables_cell_id_key, "nucleus_id"]].dropna(subset=["nucleus_id"]).copy()
    match_df["nucleus_id"] = match_df["nucleus_id"].astype(gdf_nuc.index.dtype, copy=False)

    nuc_centroids = pd.DataFrame(
        {
            "nucleus_id": gdf_nuc.index,
            f"{points_x_key}_centroid": gdf_nuc.geometry.centroid.x.to_numpy(),
            f"{points_y_key}_centroid": gdf_nuc.geometry.centroid.y.to_numpy(),
        }
    )

    centroids_df = match_df.merge(nuc_centroids, on="nucleus_id", how="left").set_index(tables_cell_id_key)

    boundary_gdf = match_df.merge(gdf_nuc, left_on="nucleus_id", right_index=True, how="left").set_index(
        tables_cell_id_key
    )[["geometry"]]

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
