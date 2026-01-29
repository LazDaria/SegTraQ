import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import _looks_like_counts, merge_into_obs
from .utils import (
    _compute_ncvs_within_radius,
    _get_center_border_counts,
    _join_points_regions,
    _norm_log_df,
    _process_cell,
)


def compute_cell_nuc_match(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
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
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score (e.g. fully inside the cell), the
        larger nucleus (by area) is selected.
    min_intersection_area : float, default=0.0
        Minimum area(cell ∩ nucleus) required to consider a nucleus as a candidate.
        Overlaps <= this threshold are ignored.
    n_jobs : int, optional
        Number of parallel jobs. Default=-1 uses all CPUs.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
    """
    assert nucleus_shapes_key is not None, (
        "Cannot compute IoUs: `nucleus_shapes_key` is None. "
        "Define a valid nucleus shape layer in the `SegTraQ` constructor before running `nc` metrics."
    )

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

    # Parallel loop over cells
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(_process_cell)(
            cell_row=cell_row,
            nucleus_shapes=nuc_boundaries,
            id_name=cell_boundaries.index.name,
            nuc_sindex=nuc_sindex,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
        )
        for _, cell_row in cell_boundaries.iterrows()
    )

    match_df = pd.DataFrame(results)

    # if a nucleus is assigned to multiple cells, we keep only the one with the highest fraction / IoU
    cols = (
        ["best_nuc_id", "nucleus_fraction", "IoU"]
        if select_by == "nucleus_fraction"
        else ["best_nuc_id", "IoU", "nucleus_fraction"]
    )
    match_df.loc[match_df.sort_values(cols, ascending=[True, False, False]).duplicated("best_nuc_id"), cols] = np.nan

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=match_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=cell_boundaries.index.name,
        )

    return match_df


def compute_cell_nuc_correlation(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    min_transcripts: int = 10,
    min_genes: int = 5,
    metric: str = "cosine_sim",
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
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
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    min_transcripts : int, default=10
        Minimum number of transcripts (raw counts) required per region (cell and nucleus) to compute a correlation.
        If either region has fewer than `min_transcripts` counts, the correlation is set to NaN.
    min_genes : int, default=5
        Minimum number of non-zero genes required to compute a correlation.
        If fewer genes are available, the correlation is set to NaN.
    metric : str, default="cosine_sim"
        Correlation metric to use ("pearson", "spearman", "cosine_sim" currently supported).
    n_jobs: int
        Number of jobs for computing cell nucleus match, if not yet calculated.
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score (e.g. fully inside the cell), the
        larger nucleus (by area) is selected.
    min_intersection_area : float, default=0.0
        Minimum area(cell ∩ nucleus) required to consider a nucleus as a candidate.
        Overlaps <= this threshold are ignored.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - cell_id_key : identifier of each cell,
            - `best_nuc_id`: matching nucleus ID with highest nucleus fraction or IoU (or None),
            - `corr_nc_cell`: Pearson correlation between the cell and its matched nucleus gene counts
            (NaN if no match).
    """
    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported")

    T_cells = sd.transformations.get_transformation(sdata.shapes[shapes_key])
    T_nuclei = sd.transformations.get_transformation(sdata.shapes[nucleus_shapes_key])
    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    id_key = sdata[shapes_key].index.name
    tbl = sdata.tables[tables_key]

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
            inplace=inplace,
        )
    else:
        match_df = tbl.obs[[id_key, "best_nuc_id", "IoU", "nucleus_fraction"]].copy()

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
        columns=sdata.tables[tables_key].var_names,
    )

    _, expr_nucleus = _join_points_regions(
        sdata=sdata,
        region_key=nucleus_shapes_key,
        tables_key=tables_key,
        points_key=points_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        predicate="intersects",
        require_points_region_ID_match=False,
    )

    common_genes = expr_nucleus.columns.intersection(expr_cells.columns)
    expr_nucleus = expr_nucleus[common_genes]
    expr_cells = expr_cells[common_genes]

    expr_cells_norm = _norm_log_df(expr_cells)
    expr_nucleus_norm = _norm_log_df(expr_nucleus)

    rows = []
    for _, row in match_df.iterrows():
        cid, nid = row[id_key], row.best_nuc_id
        if pd.isna(nid):  # if no overlapping nucleus
            rows.append(
                {
                    id_key: cid,
                    "best_nuc_id": nid,
                    "IoU": row.IoU,
                    "nucleus_fraction": row.nucleus_fraction,
                    "corr_nc_cell": np.nan,
                }
            )
        else:
            x_raw = expr_cells.loc[cid, :].to_numpy()
            y_raw = expr_nucleus.loc[nid, :].to_numpy()

            x_norm = expr_cells_norm.loc[cid, :].to_numpy()
            y_norm = expr_nucleus_norm.loc[nid, :].to_numpy()

            mask = (x_raw != 0) | (y_raw != 0)
            x = x_norm[mask]
            y = y_norm[mask]

            x_counts = x_raw[mask].sum()
            y_counts = y_raw[mask].sum()

            if (mask.sum() < min_genes) or (x_counts < min_transcripts) or (y_counts < min_transcripts):
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
                    "nucleus_fraction": row.nucleus_fraction,
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

    return corr_df


def compute_correlation_between_parts(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    min_transcripts: int = 10,
    min_genes: int = 5,
    metric: str = "cosine_sim",
    scale: float = 1e4,
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
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
    min_transcripts : int, default=10
        Minimum number of transcripts (raw counts) required per region (cytoplasm and nucleus) to compute a correlation.
        If either region has fewer than `min_transcripts` counts, the correlation is set to NaN.
    min_genes : int, default=5
        Minimum number of non-zero genes required to compute a correlation.
        If fewer genes are available, the correlation is set to NaN.
    metric : str, default="cosine_sim"
        Correlation metric to use ("pearson", "spearman", "cosine_sim" currently supported).
    scale: float, default=1e4,
        Scale for library size normalization.
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score (e.g. fully inside the cell), the
        larger nucleus (by area) is selected.
    min_intersection_area : float, default=0.0
        Minimum area(cell ∩ nucleus) required to consider a nucleus as a candidate.
        Overlaps <= this threshold are ignored.
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
    id_key = cells_gdf.index.name

    if "best_nuc_id" not in sdata.tables[tables_key].obs.columns:
        match_df = compute_cell_nuc_match(
            sdata=sdata,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            min_intersection_area=min_intersection_area,
            select_by=select_by,
            n_jobs=n_jobs,
            inplace=inplace,
        )
    else:
        match_df = sdata.tables[tables_key].obs[[id_key, "best_nuc_id", "IoU", "nucleus_fraction"]].copy()

    best_nuc_map = match_df.set_index(id_key)["best_nuc_id"]

    tx_cell, _ = _join_points_regions(
        sdata=sdata,
        region_key=shapes_key,
        tables_key=tables_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        predicate="within",
        require_points_region_ID_match=True,  # <-- keeps only points within their labeled cell
    )

    tx_nuc, _ = _join_points_regions(
        sdata=sdata,
        region_key=nucleus_shapes_key,
        tables_key=tables_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        predicate="within",
        require_points_region_ID_match=False,
    )

    # keep only points that were inside their assigned cell
    valid_point_ids = set(tx_cell["point_id"])
    tx = tx_nuc[tx_nuc["point_id"].isin(valid_point_ids)].copy()

    tx["best_nuc_id"] = tx[points_cell_id_key].map(best_nuc_map)
    tx["in_intersection"] = tx["region_id"].eq(tx["best_nuc_id"])

    all_cells = pd.Index(sdata.tables[tables_key].obs[tables_cell_id_key])
    all_genes = pd.Index(sdata.tables[tables_key].var_names)

    # intersection: cell ∩ best nucleus
    counts_intersection_raw = (
        tx[tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_cells, columns=all_genes, fill_value=0)
    )

    # remainder: rest of the cell
    counts_remainder_raw = (
        tx[~tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_cells, columns=all_genes, fill_value=0)
    )

    # normalize
    total_counts = (counts_intersection_raw + counts_remainder_raw).sum(axis=1).replace(0, np.nan)
    counts_intersection_norm = counts_intersection_raw.div(total_counts, axis=0) * scale
    counts_remainder_norm = counts_remainder_raw.div(total_counts, axis=0) * scale
    counts_intersection_norm = np.log1p(counts_intersection_norm).fillna(0.0)
    counts_remainder_norm = np.log1p(counts_remainder_norm).fillna(0.0)

    rows = []
    for cid in all_cells:
        if cid == 70082:
            x = 0
        nid = best_nuc_map.get(cid)
        if pd.isna(nid):  # if no overlapping nucleus
            r = np.nan
        else:
            x_raw = counts_intersection_raw.loc[cid].to_numpy(dtype=float)
            y_raw = counts_remainder_raw.loc[cid].to_numpy(dtype=float)

            # keep genes that are non-zero in at least one part
            mask = (x_raw != 0) | (y_raw != 0)

            x_counts = x_raw[mask].sum()
            y_counts = y_raw[mask].sum()

            x = counts_intersection_norm.loc[cid].to_numpy(dtype=float)[mask]
            y = counts_remainder_norm.loc[cid].to_numpy(dtype=float)[mask]

            if (mask.sum() < min_genes) or (x_counts < min_transcripts) or (y_counts < min_transcripts):
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

    out = match_df.reset_index(drop=True).merge(corr_per_cell, left_on=id_key, right_index=True, how="left")

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
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    erosion_fraction_of_radius: float = 0.2,
    radius_factor: float = 2.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
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
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
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
    min_transcripts : int, default=10
        Minimum number of transcripts (raw counts) required per region to compute a correlation.
        If either region has fewer than `min_transcripts` counts, the correlation is set to NaN.
    min_genes : int, default=5
        Minimum number of non-zero genes required to compute a correlation.
        If fewer genes are available, the correlation is set to NaN.
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
    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported")

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata,
        tables_key=tables_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    # NCV: neighborhood composition vector
    expr_ncv_raw = _compute_ncvs_within_radius(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        radius_factor=radius_factor,
    )

    common_cells = expr_border_raw.index.intersection(expr_center_raw.index)
    expr_center_raw = expr_center_raw.loc[common_cells, expr_ncv_raw.columns]
    expr_border_raw = expr_border_raw.loc[common_cells, expr_ncv_raw.columns]
    expr_ncv_raw = expr_ncv_raw.loc[common_cells, :]

    # normalization and log1p
    expr_center = _norm_log_df(expr_center_raw)
    expr_border = _norm_log_df(expr_border_raw)
    expr_ncv = _norm_log_df(expr_ncv_raw)

    id_key = sdata.shapes[shapes_key].index.name

    rows = []

    for cid in expr_center.index:
        x_center = expr_center.loc[cid].to_numpy()
        x_border = expr_border.loc[cid].to_numpy()
        x_ncv = expr_ncv.loc[cid].to_numpy()

        x_center_raw = expr_center_raw.loc[cid].to_numpy()
        x_border_raw = expr_border_raw.loc[cid].to_numpy()
        x_ncv_raw = expr_ncv_raw.loc[cid].to_numpy()

        # Filter out genes that are zero in all three regions
        mask = (x_center_raw != 0) | (x_border_raw != 0) | (x_ncv_raw != 0)
        x_center = x_center[mask]
        x_border = x_border[mask]
        x_ncv = x_ncv[mask]

        corr_center_border = np.nan
        corr_border_ncv = np.nan
        corr_ncv_vs_center = np.nan

        x_center_counts = x_center_raw[mask].sum()
        x_border_counts = x_border_raw[mask].sum()
        x_ncv_counts = x_ncv_raw[mask].sum()

        # center–border similarity
        if (mask.sum() >= min_genes) and (x_center_counts >= min_transcripts) and (x_border_counts >= min_transcripts):
            if metric == "pearson":
                corr_center_border, _ = pearsonr(x_center, x_border)
            elif metric == "spearman":
                corr_center_border, _ = spearmanr(x_center, x_border)
            elif metric == "cosine_sim":
                corr_center_border = cosine_similarity(x_center.reshape(1, -1), x_border.reshape(1, -1))[0, 0]

        # border–NCV similarity
        if (mask.sum() >= min_genes) and (x_ncv_counts >= min_transcripts) and (x_border_counts >= min_transcripts):
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
            and not np.isclose(corr_center_border, 0.0)
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
