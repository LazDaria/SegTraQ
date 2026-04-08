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
    _null_corrected_center_border_neighborhood_one_cell,
    _chi2_center_border_one_cell,
    _fisher_freeman_halton_center_border_one_cell,
    _mixture_fit_contamination_one_cell,
)


def match_nuclei_to_cells(
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
    Computes the best-matching nucleus for each cell based on Intersection-over-Union (IoU) or
    nucleus fraction (area(cell ∩ nucleus) / area(nucleus)).

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
        "Define a valid nucleus shape layer in the `SegTraQ` constructor before running `rs` metrics."
    )

    T_cells = sdata.shapes[shapes_key].attrs["transform"]
    T_nuclei = sdata.shapes[nucleus_shapes_key].attrs["transform"]

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
        ["nucleus_id", "nucleus_fraction", "iou"]
        if select_by == "nucleus_fraction"
        else ["nucleus_id", "iou", "nucleus_fraction"]
    )

    match_df.loc[match_df.sort_values(cols, ascending=[True, False, False]).duplicated("nucleus_id"), cols] = np.nan
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=match_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=cell_boundaries.index.name,
        )

    return match_df


def similarity_nucleus_cell(
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
    and computes the similarity (cosine similarity, Pearson correlation, Spearman correlation)
    between the gene expression profiles of the whole cell and its nucleus.

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
            - `nucleus_id`: matching nucleus ID with highest nucleus fraction or Intersection over Union (or None),
            - `similarity_nucleus_cell`:
                similarity (cosine similarity, Pearson correlation, Spearman correlation)
                between the cell and its matched nucleus gene counts
            (NaN if no match).
    """
    assert nucleus_shapes_key is not None, (
        "Cannot compute IoUs: `nucleus_shapes_key` is None. "
        "Define a valid nucleus shape layer in the `SegTraQ` constructor before running `rs` metrics."
    )

    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported. Please choose from 'pearson', 'spearman', or 'cosine_sim'.")

    T_cells = sdata.shapes[shapes_key].attrs["transform"]
    T_nuclei = sdata.shapes[nucleus_shapes_key].attrs["transform"]

    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    tbl = sdata.tables[tables_key]

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
            inplace=inplace,
        )

    match_df = tbl.obs[[tables_cell_id_key, "nucleus_id", "iou", "nucleus_fraction"]].copy()
    id_key = tables_cell_id_key

    X = tbl.X
    # Check if X looks like counts
    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "counts" not in tbl.layers:
        raise ValueError(
            f"'counts' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        counts = tbl.layers["counts"]
        arr = counts.toarray() if hasattr(counts, "toarray") else counts

    expr_cells = pd.DataFrame(
        arr,
        index=sdata.tables[tables_key].obs[tables_cell_id_key],
        columns=sdata.tables[tables_key].var_names,
    )

    _, expr_nucleus = _join_points_regions(
        sdata=sdata,
        region_key=nucleus_shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
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
        # cell ID and nucleus ID
        cid, nid = row[id_key], row["nucleus_id"]
        if pd.isna(nid):  # if no overlapping nucleus
            rows.append(
                {
                    id_key: cid,
                    "nucleus_id": nid,
                    "iou": row.iou,
                    "nucleus_fraction": row.nucleus_fraction,
                    "similarity_nucleus_cell": np.nan,
                }
            )
        else:
            # x is the expression from the whole cell, y from the nucleus
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
                    "nucleus_id": nid,
                    "iou": row.iou,
                    "nucleus_fraction": row.nucleus_fraction,
                    "similarity_nucleus_cell": corr,
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


def similarity_nucleus_cytoplasm(
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
    debug_cell_id: str | None = None,
):
    """
    For each cell in the SpatialData table, identifies the nucleus with highest intersection over union (IoU)
    and computes the similarity (cosine similarity, Pearson correlation, Spearman correlation)
    between the gene expression profiles of the cytoplasm (cell - nucleus) and the cell region overlapping the nucleus.

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
        DataFrame with columns [cell_id_key, "nucleus_id", "similarity_nucleus_cytoplasm"]
    """
    assert nucleus_shapes_key is not None, (
        "Cannot compute IoUs: `nucleus_shapes_key` is None. "
        "Define a valid nucleus shape layer in `SegTraQ` before running `rs` metrics."
    )

    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported. Please choose from 'pearson', 'spearman', or 'cosine_sim'.")

    T_cells = sdata.shapes[shapes_key].attrs["transform"]
    T_nuclei = sdata.shapes[nucleus_shapes_key].attrs["transform"]

    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    cells_gdf = sdata.shapes[shapes_key]
    id_key = cells_gdf.index.name

    if "nucleus_id" not in sdata.tables[tables_key].obs.columns:
        match_df = match_nuclei_to_cells(
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
        match_df = sdata.tables[tables_key].obs[[id_key, "nucleus_id", "iou", "nucleus_fraction"]].copy()

    best_nuc_map = match_df.set_index(id_key)["nucleus_id"]

    tx_cell, _ = _join_points_regions(
        sdata=sdata,
        region_key=shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
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
        tables_cell_id_key=tables_cell_id_key,
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

    tx["nucleus_id"] = tx[points_cell_id_key].map(best_nuc_map)
    tx["in_intersection"] = tx["region_id"].eq(tx["nucleus_id"])

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

    corr_per_cell = pd.DataFrame(rows, columns=[points_cell_id_key, "similarity_nucleus_cytoplasm"]).set_index(
        points_cell_id_key
    )

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


def similarity_border_neighborhood(
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
    neighborhood_radius_factor: float = 2.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    metric: str = "cosine_sim",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Computes the similarity between gene expression profiles in the border region of each cell
    and two references: (1) the center region of the same cell, and (2) the neighborhood composition vector (NCV)
    computed within a specified radius around the cell.

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
    neighborhood_radius_factor : float, default=2.0
        For each cell, the neighborhood consists of the cells whose centroids
        lie within the radius of the cell times this factor.
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
            - `similarity_center_border`: similarity between center and border expression,
            - `similarity_border_neighborhood`: similarity between border and neighborhood expression
            - `ratio_border_neighborhood_to_center`: ratio of the two similarities. A value > 1 indicates
              that the border is more similar to the neighborhood than to the center, while a value < 1 indicates
              the opposite.
    """
    if metric not in ["pearson", "spearman", "cosine_sim"]:
        raise ValueError(f"Metric {metric} not supported. Please choose from 'pearson', 'spearman', or 'cosine_sim'.")

    assert erosion_fraction_of_radius > 0.0 and erosion_fraction_of_radius < 1.0, (
        "`erosion_fraction_of_radius` must be between 0 and 1."
    )

    assert neighborhood_radius_factor > 1.0, "`neighborhood_radius_factor` must be larger than 1.0."

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    # expression in neighborhood
    expr_neighborhood_raw = _compute_ncvs_within_radius(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )

    common_cells = expr_border_raw.index.intersection(expr_center_raw.index)
    expr_center_raw = expr_center_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_border_raw = expr_border_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_neighborhood_raw = expr_neighborhood_raw.loc[common_cells, :]

    # normalization and log1p
    expr_center = _norm_log_df(expr_center_raw)
    expr_border = _norm_log_df(expr_border_raw)
    expr_neighborhood = _norm_log_df(expr_neighborhood_raw)

    id_key = sdata.shapes[shapes_key].index.name

    rows = []

    for cid in expr_center.index:
        x_center = expr_center.loc[cid].to_numpy()
        x_border = expr_border.loc[cid].to_numpy()
        x_neighborhood = expr_neighborhood.loc[cid].to_numpy()

        x_center_raw = expr_center_raw.loc[cid].to_numpy()
        x_border_raw = expr_border_raw.loc[cid].to_numpy()
        x_neighborhood_raw = expr_neighborhood_raw.loc[cid].to_numpy()

        # for the filtering, we need to do it independently for the two comparisons
        # the reason is that some genes may be 0 in center and border, but expressed in neighborhood
        # this will lead to higher correlations, because we have more 0s in common

        # === comparing center and border ===
        mask = (x_center_raw != 0) | (x_border_raw != 0)
        x_center_filtered = x_center[mask]
        x_border_filtered = x_border[mask]

        corr_center_border = np.nan
        corr_border_neighborhood = np.nan

        x_center_counts = x_center_raw[mask].sum()
        x_border_counts = x_border_raw[mask].sum()

        # center–border similarity
        if (mask.sum() >= min_genes) and (x_center_counts >= min_transcripts) and (x_border_counts >= min_transcripts):
            if metric == "pearson":
                corr_center_border, _ = pearsonr(x_center_filtered, x_border_filtered)
            elif metric == "spearman":
                corr_center_border, _ = spearmanr(x_center_filtered, x_border_filtered)
            elif metric == "cosine_sim":
                corr_center_border = cosine_similarity(
                    x_center_filtered.reshape(1, -1), x_border_filtered.reshape(1, -1)
                )[0, 0]

        # === comparing border and neighborhood ===
        mask = (x_border_raw != 0) | (x_neighborhood_raw != 0)
        x_border_filtered = x_border[mask]
        x_neighborhood_filtered = x_neighborhood[mask]

        x_border_counts = x_border_raw[mask].sum()
        x_neighborhood_counts = x_neighborhood_raw[mask].sum()

        if (
            (mask.sum() >= min_genes)
            and (x_neighborhood_counts >= min_transcripts)
            and (x_border_counts >= min_transcripts)
        ):
            if metric == "pearson":
                corr_border_neighborhood, _ = pearsonr(x_border_filtered, x_neighborhood_filtered)
            elif metric == "spearman":
                corr_border_neighborhood, _ = spearmanr(x_border_filtered, x_neighborhood_filtered)
            elif metric == "cosine_sim":
                corr_border_neighborhood = cosine_similarity(
                    x_border_filtered.reshape(1, -1), x_neighborhood_filtered.reshape(1, -1)
                )[0, 0]

        # ratio: border–neighborhood vs center–border
        ratio_border_neighborhood_to_center = np.nan
        if (
            not np.isnan(corr_center_border)
            and not np.isnan(corr_border_neighborhood)
            and not np.isclose(corr_center_border, 0.0)
        ):
            ratio_border_neighborhood_to_center = corr_border_neighborhood / corr_center_border

        rows.append(
            {
                id_key: cid,
                "similarity_center_border": corr_center_border,
                "similarity_border_neighborhood": corr_border_neighborhood,
                "ratio_border_neighborhood_to_center": ratio_border_neighborhood_to_center,
            }
        )

    corr_df = pd.DataFrame(rows)

    # check that the df is not empty
    if corr_df.empty:
        raise ValueError(
            "Could not compute similarities. "
            "Try different parameters for erosion_fraction_of_radius or neighborhood_radius_factor. "
            "You used erosion_fraction_of_radius="
            f"{erosion_fraction_of_radius} and neighborhood_radius_factor={neighborhood_radius_factor}."
        )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=corr_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return corr_df


# custom method for debugging
def get_genes_in_compartment(
    cell,
    compartment,
    sdata,
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
    scale: float = 1e4,
    erosion_fraction_of_radius: float = 0.2,
):
    if compartment == "nuc_cyto":
        cells_gdf = sdata.shapes[shapes_key]
        id_key = cells_gdf.index.name

        if "nucleus_id" not in sdata.tables[tables_key].obs.columns:
            raise ValueError("Nucleus-cell matching has not been performed yet.")
        else:
            match_df = sdata.tables[tables_key].obs[[id_key, "nucleus_id", "iou", "nucleus_fraction"]].copy()
        best_nuc_map = match_df.set_index(id_key)["nucleus_id"]

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

        tx["nucleus_id"] = tx[points_cell_id_key].map(best_nuc_map)
        tx["in_intersection"] = tx["region_id"].eq(tx["nucleus_id"])

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

        all_genes = counts_intersection_raw.columns

        cid = cell
        nid = best_nuc_map.get(cid)
        if pd.isna(nid):  # if no overlapping nucleus
            raise ValueError(f"Cell {cid} has no overlapping nucleus.")
        else:
            x_raw = counts_intersection_raw.loc[cid].to_numpy(dtype=float)
            y_raw = counts_remainder_raw.loc[cid].to_numpy(dtype=float)

            # keep genes that are non-zero in at least one part
            mask = (x_raw != 0) | (y_raw != 0)

            genes_in_either = all_genes[mask].tolist()
            genes_in_nucleus = all_genes[mask & (x_raw != 0)].tolist()
            genes_in_cytoplasm = all_genes[mask & (y_raw != 0)].tolist()

            return {
                "genes_in_either": genes_in_either,
                "genes_in_nucleus": genes_in_nucleus,
                "genes_in_cytoplasm": genes_in_cytoplasm,
            }
    elif compartment == "center_border":
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

        common_cells = expr_border_raw.index.intersection(expr_center_raw.index)
        expr_center_raw = expr_center_raw.loc[common_cells, expr_center_raw.columns]
        expr_border_raw = expr_border_raw.loc[common_cells, expr_border_raw.columns]

        id_key = sdata.shapes[shapes_key].index.name

        cid = cell
        x_center_raw = expr_center_raw.loc[cid].to_numpy()
        x_border_raw = expr_border_raw.loc[cid].to_numpy()

        # for the filtering, we need to do it independently for the two comparisons
        # the reason is that some genes may be 0 in center and border, but expressed in neighborhood
        # this will lead to higher correlations, because we have more 0s in common

        # === comparing center and border ===
        mask = (x_center_raw != 0) | (x_border_raw != 0)
        genes_in_either = sdata.tables[tables_key].var_names[mask].tolist()
        genes_in_center = sdata.tables[tables_key].var_names[mask & (x_center_raw != 0)].tolist()
        genes_in_border = sdata.tables[tables_key].var_names[mask & (x_border_raw != 0)].tolist()

        return {
            "genes_in_either": genes_in_either,
            "genes_in_center": genes_in_center,
            "genes_in_border": genes_in_border,
        }
    else:
        raise ValueError(f"Compartment {compartment} not recognized. Use 'nuc_cyto' or 'center_border'.")

def null_corrected_center_border_similarity(
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
    neighborhood_radius_factor: float = 2.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    n_sim: int = 200,
    scale: float = 1e4,
    inplace: bool = True,
    random_state: int | None = 0,
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> pd.DataFrame:
    """
    Compute null-corrected border-related cosine similarities.

    For each cell, this yields:
    - center-border similarity
    - border-neighborhood similarity
    - empirical null mean / SD / quantiles for both comparisons
    - residuals relative to the null mean
    - a contamination score:
      similarity_border_neighborhood_residual - similarity_center_border_residual

    Null distributions are obtained via symmetric random partitioning of pooled
    counts (center+border and border+neighborhood), preserving observed totals.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for transcript x-coordinates.
    points_y_key : str, default="y"
        Column for transcript y-coordinates.
    points_gene_key : str, default="feature_name"
        Column specifying gene / feature names.
    erosion_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to erode the cell polygon and
        define the center region.
    neighborhood_radius_factor : float, default=2.0
        Radius factor used to define neighboring cells when computing the
        neighborhood count vector.
    min_transcripts : int, default=10
        Minimum total transcript count required for center, border, and
        neighborhood after restricting to the shared gene space.
    min_genes : int, default=5
        Minimum number of genes required in the shared gene space.
    n_sim : int, default=200
        Number of null simulations per cell.
    scale : float, default=1e4
        Library-size scaling factor applied before log1p.
    inplace : bool, default=True
        Whether to merge the resulting metrics into `sdata.tables[tables_key].obs`.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - `tables_cell_id_key`
            - `similarity_center_border`
            - `similarity_center_border_null_mean`
            - `similarity_center_border_null_sd`
            - `similarity_center_border_residual`
            - `similarity_center_border_zscore`
            - `similarity_border_neighborhood`
            - `similarity_border_neighborhood_null_mean`
            - `similarity_border_neighborhood_null_sd`
            - `similarity_border_neighborhood_residual`
            - `similarity_border_neighborhood_zscore`
            - `contamination_score`
            - count and gene-usage summaries
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    expr_neighborhood_raw = _compute_ncvs_within_radius(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )

    common_cells = (
        expr_center_raw.index
        .intersection(expr_border_raw.index)
        .intersection(expr_neighborhood_raw.index)
    )

    # Align rows and columns so all three matrices refer to the same cells/genes.
    expr_center_raw = expr_center_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_border_raw = expr_border_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_neighborhood_raw = expr_neighborhood_raw.loc[common_cells, :]

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=len(common_cells))

    rows = []
    for cid, seed in zip(common_cells, seeds):
        res = _null_corrected_center_border_neighborhood_one_cell(
            x_center_raw=expr_center_raw.loc[cid].to_numpy(),
            x_border_raw=expr_border_raw.loc[cid].to_numpy(),
            x_neighborhood_raw=expr_neighborhood_raw.loc[cid].to_numpy(),
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            n_sim=n_sim,
            scale=scale,
            random_state=int(seed),
            q_low=q_low,
            q_high=q_high,
        )
        res[id_key] = cid
        rows.append(res)

    out = pd.DataFrame(rows)

    if inplace and not out.empty:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out

def chi2_center_border_similarity(
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
    min_transcripts: int = 10,
    min_genes: int = 5,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Compute a per-cell chi-square test comparing center and border
    expression compositions.

    This is mainly intended as a comparison baseline for the null-corrected
    cosine similarity metric.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for transcript x-coordinates.
    points_y_key : str, default="y"
        Column for transcript y-coordinates.
    points_gene_key : str, default="feature_name"
        Column specifying gene / feature names.
    erosion_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to erode the cell polygon and
        define the center region.
    min_transcripts : int, default=10
        Minimum total transcript count required for center and border.
    min_genes : int, default=5
        Minimum number of genes required after restricting to the shared gene space.
    inplace : bool, default=True
        Whether to merge the resulting metrics into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id
            - chi2 statistic
            - p-value
            - degrees of freedom
            - Cramér's V
            - count and gene-usage summaries
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    common_cells = expr_center_raw.index.intersection(expr_border_raw.index)
    common_genes = expr_center_raw.columns.intersection(expr_border_raw.columns)

    expr_center_raw = expr_center_raw.loc[common_cells, common_genes]
    expr_border_raw = expr_border_raw.loc[common_cells, common_genes]

    rows = []
    for cid in common_cells:
        res = _chi2_center_border_one_cell(
            x_center_raw=expr_center_raw.loc[cid].to_numpy(),
            x_border_raw=expr_border_raw.loc[cid].to_numpy(),
            min_transcripts=min_transcripts,
            min_genes=min_genes,
        )
        res[id_key] = cid
        rows.append(res)

    out = pd.DataFrame(rows)

    if inplace and not out.empty:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out

def fisher_center_border_similarity(
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
    min_transcripts: int = 10,
    min_genes: int = 5,
    n_sim: int = 5000,
    inplace: bool = True,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Compute a per-cell Monte Carlo Fisher-Freeman-Halton exact test comparing
    center and border expression compositions in a 2 x G contingency table.

    This is a Fisher-style fixed-margins test intended as a comparison baseline
    to the null-corrected cosine similarity metric.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for transcript x-coordinates.
    points_y_key : str, default="y"
        Column for transcript y-coordinates.
    points_gene_key : str, default="feature_name"
        Column specifying gene / feature names.
    erosion_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to erode the cell polygon and
        define the center region.
    min_transcripts : int, default=10
        Minimum total transcript count required for center and border.
    min_genes : int, default=5
        Minimum number of genes required after restricting to genes with
        nonzero total count across center+border.
    n_sim : int, default=5000
        Number of Monte Carlo null tables sampled per cell.
    inplace : bool, default=True
        Whether to merge the resulting metrics into `sdata.tables[tables_key].obs`.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id
            - observed table log-probability under the fixed-margins null
            - Fisher-style Monte Carlo p-value
            - count and gene-usage summaries
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    common_cells = expr_center_raw.index.intersection(expr_border_raw.index)
    common_genes = expr_center_raw.columns.intersection(expr_border_raw.columns)

    expr_center_raw = expr_center_raw.loc[common_cells, common_genes]
    expr_border_raw = expr_border_raw.loc[common_cells, common_genes]

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32 - 1, size=len(common_cells))

    rows = []
    for cid, seed in zip(common_cells, seeds):
        res = _fisher_freeman_halton_center_border_one_cell(
            x_center_raw=expr_center_raw.loc[cid].to_numpy(),
            x_border_raw=expr_border_raw.loc[cid].to_numpy(),
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            n_sim=n_sim,
            random_state=int(seed),
        )
        res[id_key] = cid
        rows.append(res)

    out = pd.DataFrame(rows)

    if inplace and not out.empty:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out

def mixture_fit_contamination_score(
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
    neighborhood_radius_factor: float = 2.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute a mixture-fit contamination score per cell.

    For each cell, fit:
        p_border ~ (1 - alpha) * p_center + alpha * p_neighborhood

    on normalized gene proportions, where:
        - p_center is the center expression composition
        - p_border is the border expression composition
        - p_neighborhood is the local neighborhood composition

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for transcript x-coordinates.
    points_y_key : str, default="y"
        Column for transcript y-coordinates.
    points_gene_key : str, default="feature_name"
        Column specifying gene / feature names.
    erosion_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to erode the cell polygon and
        define the center region.
    neighborhood_radius_factor : float, default=2.0
        Radius factor used to define neighboring cells when computing the
        neighborhood count vector.
    min_transcripts : int, default=10
        Minimum total transcript count required for center, border, and neighborhood.
    min_genes : int, default=5
        Minimum number of genes required in the shared gene space.
    pseudocount : float, default=0.5
        Pseudocount used when converting counts to proportions.
    inplace : bool, default=True
        Whether to merge the resulting metrics into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - tables_cell_id_key
            - mixture_alpha_hat
            - mixture_fit_center_only_l2
            - mixture_fit_mixture_l2
            - mixture_fit_improvement_l2
            - mixture_fit_improvement_fraction
            - count and gene-usage summaries
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center_raw, expr_border_raw = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_gene_key=points_gene_key,
        erosion_fraction_of_radius=erosion_fraction_of_radius,
    )

    expr_neighborhood_raw = _compute_ncvs_within_radius(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )

    common_cells = (
        expr_center_raw.index
        .intersection(expr_border_raw.index)
        .intersection(expr_neighborhood_raw.index)
    )

    expr_center_raw = expr_center_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_border_raw = expr_border_raw.loc[common_cells, expr_neighborhood_raw.columns]
    expr_neighborhood_raw = expr_neighborhood_raw.loc[common_cells, :]

    rows = []
    for cid in common_cells:
        res = _mixture_fit_contamination_one_cell(
            x_center_raw=expr_center_raw.loc[cid].to_numpy(),
            x_border_raw=expr_border_raw.loc[cid].to_numpy(),
            x_neighborhood_raw=expr_neighborhood_raw.loc[cid].to_numpy(),
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            pseudocount=pseudocount,
        )
        res[id_key] = cid
        rows.append(res)

    out = pd.DataFrame(rows)

    if inplace and not out.empty:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out

