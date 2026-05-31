import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame

from ..utils import _get_count_matrix, _get_genes, merge_into_obs
from .utils import (
    _bootstrap_mixture_fit,
    _cosine_similarity_two_vectors,
    _get_center_border_counts,
    _get_neighborhood_counts,
    _join_points_regions,
    _match_nucleus_one_cell,
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
    parallel_backend: str = "threading",
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
        Key in `sdata.tables` for the cell-level metadata table.
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
        Minimum area (cell ∩ nucleus) required to consider a nucleus as a candidate.
        Overlaps <= this threshold are ignored.
    n_jobs : int, optional
        Number of parallel jobs. Default `-1` uses all available CPU cores.
    parallel_backend : str, optional
        Parallelization backend to use with joblib. Default is "threading".
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

    # spatial index reused across all cells for efficiency
    nuc_sindex = nuc_boundaries.sindex

    # Parallel loop over cells
    results = Parallel(n_jobs=n_jobs, verbose=0, backend=parallel_backend)(
        delayed(_match_nucleus_one_cell)(
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
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    tables_raw_counts_layer: str | None = None,
    min_transcripts: int = 10,
    min_genes: int = 5,
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    scale: float = 1e4,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the cosine similarity between whole-cell and matched nucleus
    expression profiles for each cell.

    The point estimate is computed from cell and nucleus count matrices using
    genes detected in at least one of the two regions, followed by row-wise
    library-size normalization and log1p transformation.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    tables_raw_counts_layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts.
        If a layer is specified, it must exist and contain count-like values.
    min_transcripts : int, default=10
        Minimum number of transcripts required in both cell and nucleus.
    min_genes : int, default=5
        Minimum number of non-zero genes required across cell and nucleus.
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score, the larger nucleus is selected.
    min_intersection_area : float, default=0.0
        Minimum area(cell ∩ nucleus) required to consider a nucleus as a
        candidate. Overlaps <= this threshold are ignored.
    n_jobs : int, default=-1
        Number of jobs for computing cell-nucleus matches if they have not yet
        been calculated. Default `-1` uses all available CPU cores.
    parallel_backend : str, optional
        Parallelization backend to use with joblib. Default is "threading".
    scale : float, default=1e4
        Library-size normalization scale used before log1p.
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id column
            - `nucleus_id`
            - `iou`
            - `nucleus_fraction`
            - `similarity_nucleus_cell`
    """
    assert nucleus_shapes_key is not None, (
        "Cannot compute nucleus-cell similarity: `nucleus_shapes_key` is None. "
        "Define a valid nucleus shape layer before running this metric."
    )

    T_cells = sdata.shapes[shapes_key].attrs["transform"]
    T_nuclei = sdata.shapes[nucleus_shapes_key].attrs["transform"]

    assert T_cells == T_nuclei, (
        "Cell and nucleus shapes are not aligned. Please ensure they share the same transformation."
    )

    adata = sdata.tables[tables_key]
    cells_gdf = sdata.shapes[shapes_key]
    shapes_cell_id_key = cells_gdf.index.name

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
            parallel_backend=parallel_backend,
            inplace=inplace,
        )
    else:
        match_df = sdata.tables[tables_key].obs[[tables_cell_id_key, "nucleus_id", "iou", "nucleus_fraction"]].copy()
        # need to rename the column to the id_key used in shapes for the join later
        match_df = match_df.rename(columns={tables_cell_id_key: shapes_cell_id_key})

    counts = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    arr = counts.toarray() if hasattr(counts, "toarray") else counts

    expr_cells = pd.DataFrame(
        arr,
        index=adata.obs[tables_cell_id_key],
        columns=adata.var_names,
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
        tables_gene_key=tables_gene_key,
        predicate="intersects",
        require_points_region_ID_match=False,
    )

    # ensure vectors are aligned on the same gene set
    common_genes = expr_nucleus.columns.intersection(expr_cells.columns)
    expr_nucleus = expr_nucleus[common_genes]
    expr_cells = expr_cells[common_genes]

    rows = []
    for _, row in match_df.iterrows():
        cid, nid = row[shapes_cell_id_key], row["nucleus_id"]

        if pd.isna(nid):
            sim = np.nan
        else:
            x = expr_cells.loc[cid].to_numpy()
            y = expr_nucleus.loc[nid].to_numpy()

            sim = _cosine_similarity_two_vectors(
                x,
                y,
                min_transcripts=min_transcripts,
                min_genes=min_genes,
                scale=scale,
            )

        rows.append(
            {
                tables_cell_id_key: cid,
                "nucleus_id": nid,
                "iou": row.iou,
                "nucleus_fraction": row.nucleus_fraction,
                "similarity_nucleus_cell": sim,
            }
        )

    corr_df = pd.DataFrame(rows)

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=corr_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return corr_df


def similarity_nucleus_cytoplasm(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
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
    scale: float = 1e4,
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the cosine similarity between nuclear and cytoplasmic
    expression profiles for each cell.

    The point estimate is computed from count matrices for the part of the cell
    overlapping the matched nucleus and the remaining part of the cell, using
    genes detected in at least one of the two regions, followed by row-wise
    library-size normalization and log1p transformation.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    nucleus_shapes_key : str, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    min_transcripts : int, default=10
        Minimum number of transcripts required in both nuclear and
        cytoplasmic regions.
    min_genes : int, default=5
        Minimum number of non-zero genes required across nuclear and
        cytoplasmic regions.
    scale : float, default=1e4
        Library-size normalization scale used before log1p.
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score, the larger nucleus is selected.
    min_intersection_area : float, default=0.0
        Minimum area (cell ∩ nucleus) required to consider a nucleus as a
        candidate. Overlaps <= this threshold are ignored.
    n_jobs : int, default=-1
        Number of jobs for computing cell-nucleus matches if they have not yet
        been calculated. Default `-1` uses all available CPU cores.
        parallel_backend : str, optional
        Parallelization backend to use with joblib. Default is "threading".
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id column
            - `nucleus_id`
            - `iou`
            - `nucleus_fraction`
            - `similarity_nucleus_cytoplasm`
    """
    assert nucleus_shapes_key is not None, (
        "Cannot compute nucleus-cytoplasm similarity: `nucleus_shapes_key` is None. "
        "Define a valid nucleus shape layer before running this metric."
    )

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
            parallel_backend=parallel_backend,
            inplace=inplace,
        )
    else:
        match_df = sdata.tables[tables_key].obs[[tables_cell_id_key, "nucleus_id", "iou", "nucleus_fraction"]].copy()
        # need to rename the column to the id_key used in shapes for the join later
        match_df = match_df.rename(columns={tables_cell_id_key: id_key})

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
        tables_gene_key=tables_gene_key,
        predicate="within",
        require_points_region_ID_match=True,
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
        tables_gene_key=tables_gene_key,
        predicate="within",
        require_points_region_ID_match=False,
    )

    # restrict to transcripts that belong to cells
    valid_point_ids = set(tx_cell["point_id"])
    tx = tx_nuc[tx_nuc["point_id"].isin(valid_point_ids)].copy()

    tx["nucleus_id"] = tx[points_cell_id_key].map(best_nuc_map)
    # flag transcripts inside the matched nucleus
    tx["in_intersection"] = tx["region_id"].eq(tx["nucleus_id"])

    all_cells = pd.Index(sdata.tables[tables_key].obs[tables_cell_id_key])
    all_genes = _get_genes(
        adata=sdata.tables[tables_key],
        gene_key=tables_gene_key,
    )

    counts_intersection = (
        tx[tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_cells, columns=all_genes, fill_value=0)
    )

    counts_cytoplasm = (
        tx[~tx["in_intersection"]]
        .groupby([points_cell_id_key, points_gene_key])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_cells, columns=all_genes, fill_value=0)
    )

    rows = []
    for cid in all_cells:
        nid = best_nuc_map.get(cid)

        if pd.isna(nid):
            sim = np.nan
        else:
            x = counts_intersection.loc[cid].to_numpy(dtype=float)
            y = counts_cytoplasm.loc[cid].to_numpy(dtype=float)

            sim = _cosine_similarity_two_vectors(
                x,
                y,
                min_transcripts=min_transcripts,
                min_genes=min_genes,
                scale=scale,
            )

        rows.append(
            {
                id_key: cid,
                "similarity_nucleus_cytoplasm": sim,
            }
        )

    sim_df = pd.DataFrame(rows)

    out = match_df.merge(sim_df, on=id_key, how="left")

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out


def similarity_center_border(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    min_transcripts: int = 10,
    min_genes: int = 5,
    scale: float = 1e4,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the cosine similarity between the center and border expression
    profiles of each cell.

    The point estimate is computed from center and border count matrices
    obtained via `_get_center_border_counts`, using row-wise library-size
    normalization followed by log1p.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    border_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to define the thickness of the
        border region (outer ring).
    buffer_fraction_of_radius : float, default=0.1
        Additional fraction of the equivalent radius used to define the gap
        between the border and center regions.
    min_transcripts : int, default=10
        Minimum number of transcripts required in both center and border.
    min_genes : int, default=5
        Minimum number of non-zero genes required across center and border.
    scale : float, default=1e4
        Library-size normalization scale used before log1p.
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id column
            - `similarity_center_border`
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center, expr_border = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        tables_gene_key=tables_gene_key,
        border_fraction_of_radius=border_fraction_of_radius,
        buffer_fraction_of_radius=buffer_fraction_of_radius,
    )

    all_cells = pd.Index(sdata.tables[tables_key].obs[tables_cell_id_key])
    all_genes = _get_genes(
        adata=sdata.tables[tables_key],
        gene_key=tables_gene_key,
    )

    # ensure all cells/genes are present even if zero counts
    expr_center = expr_center.reindex(index=all_cells, columns=all_genes, fill_value=0)
    expr_border = expr_border.reindex(index=all_cells, columns=all_genes, fill_value=0)

    rows = []
    for cid in all_cells:
        x_center = expr_center.loc[cid].to_numpy()
        x_border = expr_border.loc[cid].to_numpy()

        sim = _cosine_similarity_two_vectors(
            x_center,
            x_border,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            scale=scale,
        )

        rows.append(
            {
                id_key: cid,
                "similarity_center_border": sim,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "Could not compute center-border cosine similarities. "
            "Try different parameters for border_fraction_of_radius. "
            f"You used {border_fraction_of_radius=}."
        )

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
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    neighborhood_radius_factor: float = 1.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    scale: float = 1e4,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the cosine similarity between the border and neighborhood expression
    profiles of each cell.

    The point estimate is computed from the border count matrix obtained via
    `_get_center_border_counts` and the neighborhood count matrix obtained via
    `_get_neighborhood_counts`, using row-wise library-size normalization
    followed by log1p.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned
        spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` for spot/transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript/spot.
    border_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to define the thickness of the
        border region.
    buffer_fraction_of_radius : float, default=0.1
        Additional fraction of the equivalent radius used to define the gap
        between the border and center regions.
    neighborhood_radius_factor : float, default=1.0
        Neighbor distance threshold used by `_get_neighborhood_counts`.
    min_transcripts : int, default=10
        Minimum number of transcripts required in both border and neighborhood.
    min_genes : int, default=5
        Minimum number of non-zero genes required across border and neighborhood.
    scale : float, default=1e4
        Library-size normalization scale used before log1p.
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - cell id column
            - `similarity_border_neighborhood`
    """
    id_key = sdata.shapes[shapes_key].index.name

    _, expr_border = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        tables_gene_key=tables_gene_key,
        border_fraction_of_radius=border_fraction_of_radius,
        buffer_fraction_of_radius=buffer_fraction_of_radius,
    )

    # neighborhood expression aggregated from nearby cells
    expr_neighborhood, _n_neighbors = _get_neighborhood_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        tables_gene_key=tables_gene_key,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )

    all_cells = pd.Index(sdata.tables[tables_key].obs[tables_cell_id_key])
    all_genes = _get_genes(
        adata=sdata.tables[tables_key],
        gene_key=tables_gene_key,
    )

    expr_border = expr_border.reindex(index=all_cells, columns=all_genes, fill_value=0)
    expr_neighborhood = expr_neighborhood.reindex(index=all_cells, columns=all_genes, fill_value=0)

    rows = []
    for cid in all_cells:
        x_border = expr_border.loc[cid].to_numpy()
        x_neighborhood = expr_neighborhood.loc[cid].to_numpy()

        sim = _cosine_similarity_two_vectors(
            x_border,
            x_neighborhood,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            scale=scale,
        )

        rows.append({id_key: cid, "similarity_border_neighborhood": sim})

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "Could not compute border-neighborhood cosine similarities. "
            "Try different parameters for border_fraction_of_radius or "
            "neighborhood_radius_factor. You used "
            f"{border_fraction_of_radius=} and {neighborhood_radius_factor=}."
        )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out


def border_admixture_score(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    neighborhood_radius_factor: float = 1.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
    n_boot: int = 200,
    ci_level: float = 0.95,
    random_state: int | None = None,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute a per-cell border admixture score with bootstrap confidence
    intervals.

    For each focal cell, transcripts are grouped into center, border, and
    neighborhood regions. The border profile is compared against a center-only
    model and a fitted center-neighborhood mixture model. The reported score is
    the relative improvement of the mixture model over the center-only model.

    Bootstrap confidence intervals are obtained by multinomial resampling of
    the observed per-region gene count vectors within each focal cell.

    Notes
    -----
    - The center and border count matrices are computed using
      `_get_center_border_counts`.
    - Neighborhood counts are computed by summing transcript count vectors of
      neighboring cells returned by `_find_neighbors_by_distance`.

    Parameters
    ----------
    sdata
        SpatialData object.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table containing cell ids.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` containing cell polygons.
    points_key : str, default="transcripts"
        Key in `sdata.points` containing transcript points.
    points_cell_id_key : str, default="cell_id"
        Column in the transcript table containing transcript-assigned cell ids.
    points_background_id : str, default="UNASSIGNED"
        Identifier used for background or unassigned transcripts.
    points_x_key : str, default="x"
        X-coordinate column in the transcript table.
    points_y_key : str, default="y"
        Y-coordinate column in the transcript table.
    points_gene_key : str, default="feature_name"
        Column containing gene names.
    border_fraction_of_radius : float, default=0.2
        Fraction of the equivalent radius used to define the thickness of the
        border region (outer ring).
    buffer_fraction_of_radius : float, default=0.1
        Additional fraction of the equivalent radius used to define the gap
        between the border and center regions.
    neighborhood_radius_factor : float, default=1.0
        Neighbor distance threshold expressed as a multiple of the focal cell's
        equivalent radius.
    min_transcripts : int, default=10
        Minimum number of transcripts required in each region.
    min_genes : int, default=5
        Minimum number of genes required across the three regions combined.
    pseudocount : float, default=0.5
        Pseudocount used when converting counts to proportions.
    n_boot : int, default=200
        Number of bootstrap replicates per cell.
    ci_level : float, default=0.95
        Percentile confidence interval level.
    random_state : int | None, default=None
        Random seed for reproducible bootstrap resampling.
    n_jobs : int, default=-1
        Number of parallel jobs across cells. Default `-1` uses all available CPU cores.
    parallel_backend : str, optional
        Parallelization backend to use with joblib. Default is "threading".
    inplace : bool, default=True
        If True, merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        One row per cell with columns:
        - cell id column
        - `border_admixture_score`
        - `border_admixture_score_ci_low`
        - `border_admixture_score_ci_high`
    """
    id_key = sdata.shapes[shapes_key].index.name

    expr_center, expr_border = _get_center_border_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        tables_gene_key=tables_gene_key,
        border_fraction_of_radius=border_fraction_of_radius,
        buffer_fraction_of_radius=buffer_fraction_of_radius,
    )

    # neighborhood expression aggregated from nearby cells
    expr_neighborhood, _n_neighbors = _get_neighborhood_counts(
        sdata=sdata,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        shapes_key=shapes_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        tables_gene_key=tables_gene_key,
        neighborhood_radius_factor=neighborhood_radius_factor,
    )

    # restrict to cells with all three profiles available
    common_cells = expr_center.index.intersection(expr_border.index).intersection(expr_neighborhood.index)
    expr_center = expr_center.loc[common_cells]
    expr_border = expr_border.loc[common_cells]
    expr_neighborhood = expr_neighborhood.loc[common_cells]

    seed_rng = np.random.default_rng(random_state)

    # independent RNG seeds per cell for reproducible parallel bootstrap
    seeds = seed_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=len(common_cells),
        dtype=np.uint32,
    )

    def _bootstrap_mixture_fit_one_cell(cid, seed):
        rng = np.random.default_rng(int(seed))

        res = _bootstrap_mixture_fit(
            x_center=expr_center.loc[cid].to_numpy(dtype=int),
            x_border=expr_border.loc[cid].to_numpy(dtype=int),
            x_neighborhood=expr_neighborhood.loc[cid].to_numpy(dtype=int),
            n_boot=n_boot,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            pseudocount=pseudocount,
            ci_level=ci_level,
            rng=rng,
        )

        return {
            id_key: cid,
            "border_admixture_score": res["border_admixture_score"],
            "border_admixture_score_ci_low": res["border_admixture_score_ci_low"],
            "border_admixture_score_ci_high": res["border_admixture_score_ci_high"],
        }

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_bootstrap_mixture_fit_one_cell)(cid, seed) for cid, seed in zip(common_cells, seeds, strict=False)
    )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "Could not compute border admixture scores. "
            "Try different parameters for border_fraction_of_radius or neighborhood_radius_factor. "
            "You used "
            f"{border_fraction_of_radius=} and {neighborhood_radius_factor=}."
        )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=id_key,
        )

    return out
