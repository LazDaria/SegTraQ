import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import pearsonr
from tqdm import tqdm

from ..utils import _looks_like_counts, merge_into_obs
from .utils import _nucleus_by_feature_df, _process_cell


def compute_cell_nuc_ious(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    n_jobs: int = -1,
    use_progress: bool = True,
    inplace: bool = True,
) -> DataFrame:
    """
    Compute per-cell IoU between cell and nucleus boundaries in a SpatialData object.

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
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
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
    T_cells = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    T_nuclei = sd.transformations.get_transformation(sdata.shapes[nucleus_shapes_key])
    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    # Get GeoDataFrames
    cell_boundaries = sdata.shapes[shapes_key]
    nuc_boundaries = sdata.shapes[nucleus_shapes_key]

    # Build spatial index once
    nuc_sindex = nuc_boundaries.sindex

    # Iterator for cells
    iterator = cell_boundaries.iterrows()
    if use_progress:
        iterator = tqdm(
            iterator,
            total=len(cell_boundaries),
            desc="Processing IoU between cells and nuclei",
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
            nuc_boundaries=nuc_boundaries,
            nuc_sindex=nuc_sindex,
        )
        for _, cell_row in iterator
    )

    iou_df = pd.DataFrame(results)

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=iou_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return iou_df


def compute_cell_nuc_correlation(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str | None = "z",
    metric: str = "pearson",
    n_jobs_iou: int = -1,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    For each cell in the SpatialData table, identifies the nucleus with highest IoU
    and computes a correlation (e.g. Pearson) between the gene expression profiles
    of the cell and that nucleus.

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
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_z_key : str or None, optional, default="z"
        Column for the z-coordinate (3D data). If `None`, data are treated as 2D.
    n_jobs_iou: int
        Number of jobs for computing IoU, if not yet calculated.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - cell_id_key : identifier of each cell,
            - `best_nuc_id`: matching nucleus ID with highest IoU (or None),
            - `corr_nc_cell`: Pearson correlation between the cell and its matched nucleus gene counts
            (0.0 if no match).
    """

    T_cells = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    T_nuclei = sd.transformations.get_transformation(sdata.shapes[nucleus_shapes_key])
    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
    elif sdata[shapes_key].index.name is not None:
        id_key = sdata[shapes_key].index.name
    else:
        id_key = tables_cell_id_key

    tbl = sdata.tables[tables_key]

    if "best_nuc_id" not in tbl.obs.columns:
        iou_df = compute_cell_nuc_ious(
            sdata=sdata,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            shapes_cell_id_key=shapes_cell_id_key,
            nucleus_shapes_key=nucleus_shapes_key,
            n_jobs=n_jobs_iou,
            inplace=inplace,
        )
    else:
        iou_df = tbl.obs[[id_key, "best_nuc_id", "IoU"]].copy()

    X = tbl.X
    # Check if X looks like counts
    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in tbl.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = tbl.layers["raw"]
        arr = raw.toarray() if hasattr(raw, "toarray") else raw

    expr_cells = pd.DataFrame(
        arr,
        index=sdata.tables[tables_key].obs[tables_cell_id_key],
        columns=sdata.tables[
            tables_key
        ].var.index,  # TODO - this might break, if var.index and points_gene_key do not match!
    )

    expr_nucleus_df = _nucleus_by_feature_df(
        sdata, points_key, nucleus_shapes_key, points_gene_key, points_x_key, points_y_key, points_z_key
    )

    common_genes = expr_nucleus_df.columns.intersection(expr_cells.columns)
    expr_nucleus = expr_nucleus_df[common_genes]
    expr_cells = expr_cells[common_genes]

    rows = []
    for _, row in iou_df.iterrows():
        cid, nid = row[id_key], row.best_nuc_id
        if pd.isna(nid):  # if no overlapping nucleus
            rows.append(
                {
                    id_key: cid,
                    "best_nuc_id": np.nan,
                    "IoU": row.IoU,
                    "corr_nc_cell": 0.0,
                }
            )
        else:
            x = expr_cells.loc[cid, :].to_numpy().ravel()
            y = expr_nucleus.loc[nid, :].to_numpy().ravel()
            if metric == "pearson":
                corr, _ = pearsonr(x, y)
            else:
                raise ValueError(f"Metric {metric} not supported")  # TODO
            rows.append(
                {
                    id_key: cid,
                    "best_nuc_id": nid,
                    "IoU": row.IoU,
                    "corr_nc_cell": corr,
                }
            )

    corr_df = pd.DataFrame(rows)

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=corr_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return corr_df, iou_df


def _pearson_corr_parts(mat: pd.DataFrame) -> pd.DataFrame:
    # 1) Move "part" from index to columns
    mat_unstack = mat.unstack("part")  # index: cell_id, columns: (gene, part)

    # 2) Extract the two matrices (one row per cell, one col per gene)
    # This will create NaNs where a cell is missing that part.
    intersection = mat_unstack.xs("intersection", level="part", axis=1)
    remainder = mat_unstack.xs("remainder", level="part", axis=1)

    # 3) Convert to NumPy
    X = intersection.to_numpy(dtype=float)
    Y = remainder.to_numpy(dtype=float)

    # 4) Mask out rows where intersection or remainder is entirely zero or missing
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1) & (X.sum(axis=1) != 0) & (Y.sum(axis=1) != 0)

    # Prepare result array filled with NaNs
    corr = np.full(X.shape[0], np.nan, dtype=float)

    # 5) Compute Pearson correlation row-wise for valid rows only
    Xv = X[valid]
    Yv = Y[valid]

    # subtract row means
    Xc = Xv - Xv.mean(axis=1, keepdims=True)
    Yc = Yv - Yv.mean(axis=1, keepdims=True)

    num = (Xc * Yc).sum(axis=1)
    den = np.sqrt((Xc**2).sum(axis=1) * (Yc**2).sum(axis=1))

    # avoid division by zero
    nonzero = den != 0
    corr_valid = np.full(Xv.shape[0], np.nan, dtype=float)
    corr_valid[nonzero] = num[nonzero] / den[nonzero]

    corr[valid] = corr_valid

    # 6) Wrap back into a Series / DataFrame
    corr_per_cell = pd.Series(corr, index=mat_unstack.index, name="correlation_parts").to_frame()

    return corr_per_cell


def compute_correlation_between_parts(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    n_jobs: int = 1,  # joblib not strictly needed; most win is from vectorization
    inplace: bool = True,
):
    """
    Vectorized version: computes Pearson correlation between the cell∩best_nucleus
    ("intersection") and the rest of the cell ("remainder") using spatial joins.
    Returns DataFrame with columns ["cell_id", "best_nuc_id", "IoU", "correlation_parts"].

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
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
        Column name for y coordinate.
    n_jobs : int
        Number of parallel jobs for correlation computation.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [cell_id_key, "best_nuc_id", "correlation_parts"]
    """
    T_cells = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    T_nuclei = sd.transformations.get_transformation(sdata.shapes[nucleus_shapes_key])
    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    cells_gdf = sdata.shapes[shapes_key].copy()
    nucs_gdf = sdata.shapes[nucleus_shapes_key]

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
    elif cells_gdf.index.name is not None:
        id_key = cells_gdf.index.name
        cells_gdf[id_key] = cells_gdf.index
    else:
        id_key = tables_cell_id_key
        cells_gdf[id_key] = cells_gdf.index

    if "best_nuc_id" not in sdata.tables[tables_key].obs.columns:
        iou_df = compute_cell_nuc_ious(
            sdata=sdata,
            shapes_cell_id_key=shapes_cell_id_key,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            n_jobs=n_jobs,
            inplace=inplace,
        )
    else:
        iou_df = sdata.tables[tables_key].obs[[id_key, "best_nuc_id", "IoU"]].copy()

    best_nuc_map = iou_df.set_index(id_key)["best_nuc_id"]

    transcripts = sdata.points[points_key].compute()

    # subset to transcripts assigned to cells
    transcripts_df = transcripts[transcripts[points_cell_id_key] != points_background_id].copy()
    # subset to valid genes
    valid_features = pd.Index(
        sdata.tables[tables_key].var_names
    )  # TODO - this might break, if var.index and points_gene_key do not match!
    # e.g. one is Ensemble key and one is gene_key
    transcripts_df = transcripts_df.dropna(subset=[points_gene_key])
    transcripts_df = transcripts_df[transcripts_df[points_gene_key].isin(valid_features)]
    transcripts_df[points_gene_key] = transcripts_df[points_gene_key].cat.remove_unused_categories()

    tx_in_cell = transcripts_df[[points_gene_key, points_cell_id_key]]

    # Choose a single CRS (cells' CRS), and reproject other layers if needed - TODO
    target_crs = nucs_gdf.crs
    # if nucs_gdf.crs != target_crs:
    #    nucs_gdf = nucs_gdf.to_crs(target_crs)
    # transcripts -> GeoDataFrame
    transcripts_gdf = gpd.GeoDataFrame(
        transcripts_df,
        geometry=gpd.points_from_xy(transcripts_df[points_x_key], transcripts_df[points_y_key]),
        crs=transcripts_df.attrs.get("crs", target_crs) or target_crs,
    )
    # if transcripts_gdf.crs != target_crs:
    #     transcripts_gdf = transcripts_gdf.to_crs(target_crs)

    nucs_gdf.index.name = "nuc_id"

    tx_in_nuc = gpd.sjoin(
        transcripts_gdf[["geometry"]],
        nucs_gdf[["geometry"]],
        how="left",
        predicate="within",
    )[["nuc_id"]]

    tx = tx_in_cell.join(tx_in_nuc, how="left")

    tx["best_nuc_id"] = tx[points_cell_id_key].map(best_nuc_map)
    tx["in_intersection"] = (tx["nuc_id"].notna()) & (tx["nuc_id"] == tx["best_nuc_id"])
    tx["part"] = np.where(tx["in_intersection"], "intersection", "remainder")

    mat = pd.crosstab([tx[points_cell_id_key], tx["part"]], tx[points_gene_key]).fillna(0)

    # either use _pearson_corr_parts vectorized function or the slower per-cell apply (commented out below)
    corr_per_cell = _pearson_corr_parts(mat)
    # def _corr_two_cols(df_cell):
    #     df = df_cell.copy()
    #     df.index = df.index.get_level_values(1)
    #     if "intersection" not in df.index or "remainder" not in df.index:
    #         return np.nan
    #     x = df.loc["intersection"].to_numpy(dtype=float)
    #     y = df.loc["remainder"].to_numpy(dtype=float)
    #     if x.sum() == 0 or y.sum() == 0:
    #         return np.nan
    #     r, _ = pearsonr(x, y)
    #     return r

    # corr_per_cell = mat.groupby(level=0, sort=False).apply(_corr_two_cols).rename("correlation_parts").to_frame()

    out = iou_df.reset_index(drop=True).merge(corr_per_cell, left_on=id_key, right_index=True, how="left")

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out
