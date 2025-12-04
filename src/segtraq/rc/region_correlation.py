import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..utils import _looks_like_counts, merge_into_obs
from .utils import (
    _align_expression_dfs,
    _assign_nuc_to_transcripts,
    _assign_transcripts_to_center_or_border,
    _compute_ncvs_within_radius,
    _norm_log_df,
    _process_cell,
    _shapes_by_feature_df,
)


def compute_cell_nuc_ious(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str | None = None,
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
    nucleus_shapes_cell_id_key : str or None, optional, default=None
        Column linking nucleus polygons to cell IDs. If `None` is provided,
        the shape index is used as the cell ID.
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
            nucleus_shapes=nuc_boundaries,
            nucleus_shapes_cell_id_key=nucleus_shapes_cell_id_key,
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
    shapes_cell_id_key: str | None = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str | None = None,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    metric: str = "cosine_sim",
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
    shapes_cell_id_key : str or None, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
    nucleus_shapes_cell_id_key : str or None, optional, default=None
        Column linking nucleus polygons to cell IDs. If `None` but
        `nucleus_shapes_key` is provided, the shape index is used as the cell ID.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    metric : str, default="cosine_sim"
        Correlation metric to use ("pearson", "spearman", "cosine_sim" currently supported).
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
    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported")

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
            nucleus_shapes_cell_id_key=nucleus_shapes_cell_id_key,
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

    expr_nucleus = _shapes_by_feature_df(
        sdata=sdata,
        tables_cell_id_key=tables_cell_id_key,
        region_key=nucleus_shapes_key,
        region_cell_id_key=nucleus_shapes_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
    )

    common_genes = expr_nucleus.columns.intersection(expr_cells.columns)
    expr_nucleus = expr_nucleus[common_genes]
    expr_cells = expr_cells[common_genes]

    expr_cells_norm = _norm_log_df(expr_cells)
    expr_nucleus_norm = _norm_log_df(expr_nucleus)

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
            x_raw = expr_cells.loc[cid, :].to_numpy().ravel()
            y_raw = expr_nucleus.loc[nid, :].to_numpy().ravel()

            x_norm = expr_cells_norm.loc[cid, :].to_numpy().ravel()
            y_norm = expr_nucleus_norm.loc[nid, :].to_numpy().ravel()

            mask = (x_raw != 0) | (y_raw != 0)
            x = x_norm[mask]
            y = y_norm[mask]

            if np.all(x == 0) or np.all(y == 0):
                corr = np.nan
            else:
                if metric == "pearson":
                    corr, _ = pearsonr(x, y)
                elif metric == "spearman":
                    corr, _ = spearmanr(x, y)
                elif metric == "cosine_sim":
                    corr = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
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


def compute_correlation_between_parts(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    nucleus_shapes_key: str = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str | None = None,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    metric: str = "cosine_sim",
    scale: float = 1e4,
    n_jobs: int = 1,  # joblib not strictly needed; most win is from vectorization
    inplace: bool = True,
):
    """
    Vectorized version: computes Cosine similarity between the cell ∩ best_nucleus
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
    nucleus_shapes_cell_id_key : str or None, optional, default=None
        Column linking nucleus polygons to cell IDs. If `None` but
        `nucleus_shapes_key` is provided, the shape index is used as the cell ID.
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
    metric : str, default="cosine_sim"
        Correlation metric to use ("pearson", "spearman", "cosine_sim" currently supported).
    scale: float, default=1e4,
        Scale for library size normalization.
    n_jobs : int
        Number of parallel jobs for correlation computation.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [cell_id_key, "best_nuc_id", "correlation_parts"]
    """
    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported")

    T_cells = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    T_nuclei = sd.transformations.get_transformation(sdata.shapes[nucleus_shapes_key])
    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    cells_gdf = sdata.shapes[shapes_key]

    if shapes_cell_id_key is not None:
        id_key = shapes_cell_id_key
    elif cells_gdf.index.name is not None:
        id_key = cells_gdf.index.name
    else:
        id_key = tables_cell_id_key

    if "best_nuc_id" not in sdata.tables[tables_key].obs.columns:
        iou_df = compute_cell_nuc_ious(
            sdata=sdata,
            shapes_cell_id_key=shapes_cell_id_key,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            nucleus_shapes_cell_id_key=nucleus_shapes_cell_id_key,
            n_jobs=n_jobs,
            inplace=inplace,
        )
    else:
        iou_df = sdata.tables[tables_key].obs[[id_key, "best_nuc_id", "IoU"]].copy()

    best_nuc_map = iou_df.set_index(id_key)["best_nuc_id"]

    tx = _assign_nuc_to_transcripts(
        sdata=sdata,
        tables_key=tables_key,
        nucleus_shapes_key=nucleus_shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
    )

    tx["best_nuc_id"] = tx[points_cell_id_key].map(best_nuc_map)
    tx["in_intersection"] = (tx["nuc_id"].notna()) & (tx["nuc_id"] == tx["best_nuc_id"])

    # intersection: cell ∩ best nucleus
    counts_intersection = (
        tx[tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # remainder: rest of the cell
    counts_remainder = (
        tx[~tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    common_cells = counts_intersection.index.intersection(counts_remainder.index)
    common_genes = counts_intersection.columns.intersection(counts_remainder.columns)

    counts_intersection_raw = counts_intersection.loc[common_cells, common_genes]
    counts_remainder_raw = counts_remainder.loc[common_cells, common_genes]

    # normalize
    total_counts = (counts_intersection_raw + counts_remainder_raw).sum(axis=1).replace(0, np.nan)
    counts_intersection_norm = counts_intersection_raw.div(total_counts, axis=0) * scale
    counts_remainder_norm = counts_remainder_raw.div(total_counts, axis=0) * scale
    counts_intersection_norm = np.log1p(counts_intersection_norm).fillna(0.0)
    counts_remainder_norm = np.log1p(counts_remainder_norm).fillna(0.0)

    rows = []
    for cid in common_cells:
        x_raw = counts_intersection_raw.loc[cid].to_numpy(dtype=float)
        y_raw = counts_remainder_raw.loc[cid].to_numpy(dtype=float)

        # keep genes that are non-zero in at least one part
        mask = (x_raw != 0) | (y_raw != 0)

        x = counts_intersection_norm.loc[cid].to_numpy(dtype=float)[mask]
        y = counts_remainder_norm.loc[cid].to_numpy(dtype=float)[mask]

        if np.all(x == 0) or np.all(y == 0):
            r = np.nan
        else:
            if metric == "pearson":
                r, _ = pearsonr(x, y)
            elif metric == "spearman":
                r, _ = spearmanr(x, y)
            elif metric == "cosine_sim":
                r = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
            else:
                raise ValueError(f"Metric {metric} not supported")

        rows.append((cid, r))

    corr_per_cell = pd.DataFrame(rows, columns=[points_cell_id_key, "correlation_parts"]).set_index(points_cell_id_key)

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


def compute_center_border_ncv_correlation(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str | None = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    erosion_fraction_of_radius: float = 0.2,
    radius_factor: float = 2.0,
    metric: str = "cosine_sim",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    For each cell, compute a border similarity contamination score by (1) comparing
    gene expression in an eroded interior ("center") and a thin outer shell
    ("border"), and (2) comparing the border with the neighborhood
    composition vector (NCV).

    Specifically, the function:
      1. Erodes each cell polygon to obtain a center region.
      2. Defines the border region as the set difference between the full cell
         and its eroded center.
      3. Computes gene expression profiles for center and border.
      4. Computes the correlation between center and border expression.
      5. Computes the correlation between border expression and the
         NCV expression profile of the same cell.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data (tables, points, shapes, etc.).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    shapes_cell_id_key : str, default="cell_id"
        Column in `sdata.shapes[shapes_key]` linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    radius_factor : float, default=2.0
        Neighborhood radius factor in the same coordinate units as the shapes.
    erosion_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius to use as erosion
        Example: 0.2 means erode by 20% of the radius.
    metric : str, default="cosine_sim"
        Correlation metric to use ("pearson", "spearman", "cosine_sim" currently supported).
    inplace : bool, optional
        Whether to add the results to `sdata.tables[tables_key].obs`. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - `tables_cell_id_key`: identifier of each cell,
            - `corr_center_border`: correlation between center and border expression,
            - `corr_border_ncv`: correlation between border and NCV expression
            - `corr_ncv_vs_center`: ratio of the two correlations
    """

    _, _, expr_center, expr_border = _assign_transcripts_to_center_or_border(
        sdata,
        shapes_key=shapes_key,
        shapes_cell_id_key=shapes_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    # NCV: neighborhood composition vector
    expr_ncv = _compute_ncvs_within_radius(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        shapes_cell_id_key=shapes_cell_id_key,
        radius_factor=radius_factor,
    )

    # next, we align the three expression DataFrames to have the same cells and genes
    aligned_expression_dfs = _align_expression_dfs(
        {"expr_center": expr_center, "expr_border": expr_border, "expr_ncv": expr_ncv}, sdata, tables_key
    )

    expr_center_raw = aligned_expression_dfs["expr_center"]
    expr_border_raw = aligned_expression_dfs["expr_border"]
    expr_ncv_raw = aligned_expression_dfs["expr_ncv"]

    # normalization and log1p
    expr_center = _norm_log_df(expr_center_raw)
    expr_border = _norm_log_df(expr_border_raw)
    expr_ncv = _norm_log_df(expr_ncv_raw)

    id_key = expr_center.index.name

    rows = []

    for cid in expr_center.index:
        x_center = expr_center.loc[cid].to_numpy().ravel()
        x_border = expr_border.loc[cid].to_numpy().ravel()
        x_ncv = expr_ncv.loc[cid].to_numpy().ravel()

        x_center_raw = expr_center_raw.loc[cid].to_numpy().ravel()
        x_border_raw = expr_border_raw.loc[cid].to_numpy().ravel()
        x_ncv_raw = expr_ncv_raw.loc[cid].to_numpy().ravel()

        # Filter out genes that are zero in all three regions
        mask = (x_center_raw != 0) | (x_border_raw != 0) | (x_ncv_raw != 0)
        x_center = x_center[mask]
        x_border = x_border[mask]
        x_ncv = x_ncv[mask]

        corr_center_border = np.nan
        corr_border_ncv = np.nan
        corr_ncv_vs_center = np.nan

        if metric not in ["pearson", "spearman", "cosine_sim"]:
            raise ValueError(f"Metric {metric} not supported")

        # center–border similarity
        if not (np.all(x_center == 0) or np.all(x_border == 0)):
            if metric == "pearson":
                corr_center_border, _ = pearsonr(x_center, x_border)
            elif metric == "spearman":
                corr_center_border, _ = spearmanr(x_center, x_border)
            elif metric == "cosine_sim":
                corr_center_border = cosine_similarity(x_center.reshape(1, -1), x_border.reshape(1, -1))[0, 0]

        # border–NCV similarity
        if not (np.all(x_border == 0) or np.all(x_ncv == 0)):
            if metric == "pearson":
                corr_border_ncv, _ = pearsonr(x_border, x_ncv)
            elif metric == "spearman":
                corr_border_ncv, _ = spearmanr(x_border, x_ncv)
            elif metric == "cosine_sim":
                corr_border_ncv = cosine_similarity(x_border.reshape(1, -1), x_ncv.reshape(1, -1))[0, 0]

        # ratio: border–NCV vs center–border
        if (
            not np.isnan(corr_center_border)
            and not np.isnan(corr_border_ncv)
            and not np.isclose(corr_center_border, 0.0)  # TODO - set to very small value instead?
        ):
            corr_ncv_vs_center = corr_border_ncv / corr_center_border

            rows.append(
                {
                    id_key: cid,
                    "corr_center_border": corr_center_border,
                    "corr_border_ncv": corr_border_ncv,
                    "corr_ncv_vs_center": corr_ncv_vs_center,
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

    return corr_df
