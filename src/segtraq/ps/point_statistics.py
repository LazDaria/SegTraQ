from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd

from ..rs.region_similarity import match_nuclei_to_cells
from ..rs.utils import _get_filtered_points_df, _join_points_regions
from ..utils import merge_into_obs, xy_scale
from .utils import _fisher_pearson_sample_skew, _get_cell_geometry_lookup


def percentage_transcripts_in_compartments(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    cell_type_key: str = "transferred_cell_type",
    cell_type_query: str | list[str] | None = None,
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
    select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = 1,
    predicate: str = "intersects",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    For each cell, compute the percentage of transcripts assigned to that cell that fall in:
      - nucleus overlap region: cell ∩ matched_nucleus
      - cytoplasm region: cell - matched_nucleus (i.e. inside cell but not in nucleus overlap)
      - outside cell: not inside the assigned cell polygon

    Notes
    -----
    - Nuclei are matched to cells using `match_nuclei_to_cells` (one nucleus_id per cell).
    - A transcript is counted as "inside cell" only if it spatially joins to some cell polygon
      AND the joined polygon id equals its assigned `points_cell_id_key`.
    - Nuclear transcripts are those that join to the matched nucleus polygon for their cell.
      If no nucleus is matched for a cell, nuclear transcripts are zero by definition for that cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes : str | list[str] | None, optional
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript coordiantes on.
        If None, all genes are used.
    cell_type_key : str
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    cell_type_query : str | list[str] | None, optional
        If provided, compute the metric only for cells whose `cell_type_key` matches these label(s).
    tables_key : str, default="table"
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    shapes_key : str, default="cell_boundaries"
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    nucleus_shapes_key : str | None, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons (required if `centroid_region="nucleus"`).
    points_key : str, default="transcripts"
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str | int, default="UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    points_gene_key : str, default="feature_name"
        The key to access gene names within the transcript data. Default is "feature_name".
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    select_by : {"iou","nucleus_fraction"}, default="nucleus_fraction"
        Criterion to choose the best nucleus for each cell when `centroid_region="nucleus"`.
    min_intersection_area : float, default=0.0
        Minimum overlap area required to consider a nucleus a candidate for a cell.
    n_jobs : int, default=1
        Number of parallel jobs for cell-nucleus matching (if needed). `-1` uses all CPUs.
    predicate : str, default="intersects"
        Geometric predicate used to assign transcripts to cell or nucleus polygons during
        spatial joins (e.g. "covers" includes boundary points, "intersects" is more permissive).
    inplace : bool, default=True
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    DataFrame indexed by cell id with counts and percentages:
      n_total, n_in_cell, n_outside_cell, n_in_nucleus_overlap, n_in_cytoplasm
      pct_outside_cell, pct_nucleus, pct_cytoplasm
    """
    # transformations alignment check
    T_transcripts = sdata.points[points_key].attrs["transform"]
    T_shapes = sdata.shapes[shapes_key].attrs["transform"]

    assert np.array_equal(xy_scale(T_transcripts), xy_scale(T_shapes)), (
        "Cell shapes and transcripts are not aligned. Please ensure they share the same transformation."
    )

    tbl = sdata.tables[tables_key]

    # spatial join points -> cells, keeping ALL points for denominators
    pts_cells, _ = _join_points_regions(
        sdata=sdata,
        region_key=shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        predicate=predicate,
        require_points_region_ID_match=False,  # keep all points; define "inside assigned" afterwards
    )

    # inside the assigned cell polygon
    inside_cell = pts_cells["region_id"].notna() & (pts_cells["region_id"] == pts_cells[points_cell_id_key])

    # denominator: transcripts assigned to each cell (regardless of where they landed spatially)
    total = pts_cells.groupby(points_cell_id_key, observed=True).size().rename("n_total")

    # inside counts
    n_in_cell = pts_cells.loc[inside_cell].groupby(points_cell_id_key, observed=True).size().rename("n_in_cell")

    # ensure we have cell -> best nucleus id mapping
    if "nucleus_id" not in tbl.obs.columns:
        _ = match_nuclei_to_cells(
            sdata=sdata,
            tables_key=tables_key,
            tables_cell_id_key=tables_cell_id_key,
            shapes_key=shapes_key,
            nucleus_shapes_key=nucleus_shapes_key,
            select_by=select_by,
            min_intersection_area=min_intersection_area,
            n_jobs=n_jobs,
            inplace=True,
        )

    # map each cell_id to its best nucleus id
    cell_to_nuc = tbl.obs[[tables_cell_id_key, "nucleus_id"]].copy()
    cell_to_nuc = cell_to_nuc.dropna(subset=["nucleus_id"])

    # spatial join points -> nuclei (we need per-point nucleus id)
    pts_nuc, _ = _join_points_regions(
        sdata=sdata,
        region_key=nucleus_shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_gene_key=points_gene_key,
        points_cell_id_key=points_cell_id_key,
        points_background_id=points_background_id,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        predicate=predicate,
        require_points_region_ID_match=False,  # nucleus ids != cell ids
    )
    # pts_nuc has column "region_id" meaning nucleus id

    # combine per-point cell-join + nucleus-join on point_id
    # both join outputs contain point_id (from _join_points_regions)
    pts = (
        pts_cells[["point_id", points_cell_id_key, "region_id"]].rename(columns={"region_id": "cell_region_id"}).copy()
    )

    # Add the nucleus polygon id that the point fell into (region_id -> nuc_region_id)
    pts = pts.merge(
        pts_nuc[["point_id", "region_id"]].rename(columns={"region_id": "nuc_region_id"}),
        on="point_id",
        how="left",
    )

    # annotate best nucleus id per point (based on its assigned cell)
    pts = pts.merge(
        cell_to_nuc.rename(columns={tables_cell_id_key: points_cell_id_key}),
        on=points_cell_id_key,
        how="left",
    )

    # nuclear overlap: point must be inside its assigned cell AND inside the matched nucleus
    in_nucleus_overlap = inside_cell & pts["nuc_region_id"].notna() & (pts["nuc_region_id"] == pts["nucleus_id"])

    n_in_nucleus = (
        pts.loc[in_nucleus_overlap].groupby(points_cell_id_key, observed=True).size().rename("n_in_nucleus_overlap")
    )

    # cytoplasm: inside cell but not in nucleus overlap
    in_cytoplasm = inside_cell & (~in_nucleus_overlap)
    n_in_cyto = pts.loc[in_cytoplasm].groupby(points_cell_id_key, observed=True).size().rename("n_in_cytoplasm")

    # outside cell: assigned to cell but not inside assigned cell polygon
    n_outside = (total - n_in_cell).rename("n_outside_cell")

    # assemble output
    out = pd.concat([total, n_in_cell, n_outside, n_in_nucleus, n_in_cyto], axis=1).fillna(0)
    for c in ["n_total", "n_in_cell", "n_outside_cell", "n_in_nucleus_overlap", "n_in_cytoplasm"]:
        out[c] = out[c].astype(int)

    out["pct_outside_cell"] = np.where(out["n_total"] > 0, 100.0 * out["n_outside_cell"] / out["n_total"], np.nan)
    out["pct_nucleus"] = np.where(out["n_total"] > 0, 100.0 * out["n_in_nucleus_overlap"] / out["n_total"], np.nan)
    out["pct_cytoplasm"] = np.where(out["n_total"] > 0, 100.0 * out["n_in_cytoplasm"] / out["n_total"], np.nan)

    # generate column names
    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = genes[0] if len(genes) == 1 else f"{len(genes)}_genes"

    out = out[
        [
            "n_total",
            "n_outside_cell",
            "n_in_nucleus_overlap",
            "n_in_cytoplasm",
            "pct_outside_cell",
            "pct_nucleus",
            "pct_cytoplasm",
        ]
    ]

    out = out.rename(
        columns={
            "n_total": f"n_total_{feature}",
            "n_outside_cell": f"n_outside_cell_{feature}",
            "n_in_nucleus_overlap": f"n_in_nucleus_overlap_{feature}",
            "n_in_cytoplasm": f"n_in_cytoplasm_{feature}",
            "pct_outside_cell": f"pct_outside_cell_{feature}",
            "pct_nucleus": f"pct_nucleus_{feature}",
            "pct_cytoplasm": f"pct_cytoplasm_{feature}",
        }
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out.drop(columns=f"n_total_{feature}"),
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=points_cell_id_key,
        )

    return out


def distance_to_centroid(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    cell_type_key: str = "transferred_cell_type",
    cell_type_query: str | list[str] | None = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_key: str = "cell_area",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str | None = "nucleus_boundaries",
    centroid_region: Literal["cell", "nucleus"] = "cell",
    restrict_to_within_boundary: bool = False,
    select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = 1,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the Euclidean distance between (i) the mean transcript coordinate per cell and
    (ii) a centroid derived from either cell or nucleus shapes.

    If `centroid_region="cell"`, distances are measured to the centroid of `sdata.shapes[shapes_key]`.
    If `centroid_region="nucleus"`, each cell is first matched to a nucleus (see `select_by`,
    `min_intersection_area`) and distances are measured to that nucleus centroid. Optionally,
    transcripts can be restricted to lie within the cell boundary (`restrict_to_within_boundary=True`).

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes : str | list[str] | None, optional
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript coordiantes on.
        If None, all genes are used.
    cell_type_key : str
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    cell_type_query : str | list[str] | None, optional
        If provided, compute the metric only for cells whose `cell_type_key` matches these label(s).
    tables_key : str, default="table"
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_area_key : str, default="cell_area"
        Column in the table with cell area (used for normalization).
    points_gene_key : str, default="feature_name"
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, default="transcripts"
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str | int, default="UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key : str, default="cell_boundaries"
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    nucleus_shapes_key : str | None, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons (required if `centroid_region="nucleus"`).
    centroid_region : {"cell","nucleus"}, default="cell"
        Which shape centroid to use as the reference for distances.
    restrict_to_within_boundary : bool, default=False
        If True, keep only transcripts that fall within the cell boundary.
        Uses `covers`, so points on the boundary are included.
    select_by : {"iou","nucleus_fraction"}, default="nucleus_fraction"
        Criterion to choose the best nucleus for each cell when `centroid_region="nucleus"`.
    min_intersection_area : float, default=0.0
        Minimum overlap area required to consider a nucleus a candidate for a cell.
    n_jobs : int, default=1
        Number of parallel jobs for cell-nucleus matching (if needed). `-1` uses all CPUs.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        If `inplace=False`, returns a DataFrame containing per-cell mean transcript x/y, the chosen
        centroid x/y, raw distance, and the normalized distance column `distance_<feature>`.
        If `inplace=True`, returns the merged (two-column) DataFrame used to write into `.obs`.
    """
    # validate inputs
    if centroid_region not in ("cell", "nucleus"):
        raise ValueError(f"centroid_region={centroid_region!r} not supported. Use 'cell' or 'nucleus'.")
    if centroid_region == "nucleus" and not nucleus_shapes_key:
        raise ValueError("centroid_region='nucleus' requires `nucleus_shapes_key` to be not None.")

    # transformations alignment check (only for the shapes actually used)
    T_transcripts = sdata.points[points_key].attrs["transform"]

    T_shapes = sdata.shapes[shapes_key].attrs["transform"]
    assert np.array_equal(xy_scale(T_transcripts), xy_scale(T_shapes)), (
        "Cell shapes and transcripts are not aligned. Please ensure they share the same transformation."
    )
    if centroid_region == "nucleus":
        T_shapes = sdata.shapes[nucleus_shapes_key].attrs["transform"]
        assert np.array_equal(xy_scale(T_transcripts), xy_scale(T_shapes)), (
            "Nucleus shapes and transcripts are not aligned. Please ensure they share the same transformation."
        )

    tbl = sdata.tables[tables_key]

    transcript_df = _get_filtered_points_df(
        sdata=sdata,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
    )

    centroids, _ = _get_cell_geometry_lookup(
        sdata=sdata,
        region=centroid_region,
        shapes_key=shapes_key,
        nucleus_shapes_key=nucleus_shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        select_by=select_by,
        min_intersection_area=min_intersection_area,
        n_jobs=n_jobs,
        inplace=inplace,
    )

    # optionally restrict transcripts to be inside cell
    if restrict_to_within_boundary:
        cell_boundary = sdata.shapes[shapes_key][["geometry"]]
        # join cell geometry onto each transcript by cell id
        tmp = transcript_df.merge(
            cell_boundary,
            left_on=points_cell_id_key,
            right_index=True,
            how="inner",
        )
        if not tmp.empty:
            pts_geom = gpd.points_from_xy(tmp[points_x_key], tmp[points_y_key])
            poly = gpd.GeoSeries(tmp["geometry"], index=tmp.index)
            pt = gpd.GeoSeries(pts_geom, index=tmp.index)
            within = poly.covers(pt)
            transcript_df = tmp.loc[within, transcript_df.columns]
        else:
            raise ValueError(
                "No transcripts remain after restrict_to_within_boundary=True. "
                "Consider disabling it or verifying shapes/transforms."
            )

    # mean transcript coordinate per cell
    cell_means = (
        transcript_df.groupby(points_cell_id_key, sort=False)[[points_x_key, points_y_key]]
        .mean()
        .reset_index(drop=False)
    )

    # merge centroids (some cells may lack transcripts; inner keeps only cells with both)
    df_total = centroids.merge(
        cell_means,
        left_index=True,
        right_on=points_cell_id_key,
        how="inner",
    )

    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = genes[0] if len(genes) == 1 else f"{len(genes)}_genes"

    # euclidean distance (vectorized to reduce overhead)
    dxy = (
        df_total[[f"{points_x_key}_centroid", f"{points_y_key}_centroid"]].to_numpy()
        - df_total[[points_x_key, points_y_key]].to_numpy()
    )
    df_total[f"distance_to_{centroid_region}_centroid_{feature}"] = np.sqrt((dxy * dxy).sum(axis=1))

    # add cell area + normalize
    area_df = tbl.obs[[tables_cell_id_key, tables_area_key]]
    df_total = df_total.merge(area_df, left_on=points_cell_id_key, right_on=tables_cell_id_key, how="left")

    df_total[f"distance_to_{centroid_region}_centroid_norm_{feature}"] = df_total[
        f"distance_to_{centroid_region}_centroid_{feature}"
    ] / np.sqrt(df_total[tables_area_key])  # length scale
    df_total = df_total.reset_index(drop=True)

    if inplace:
        out = df_total[[tables_cell_id_key, f"distance_to_{centroid_region}_centroid_norm_{feature}"]].copy()
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        return out

    return df_total


def distance_to_membrane(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    cell_type_key: str = "transferred_cell_type",
    cell_type_query: str | list[str] | None = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_key: str = "cell_area",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    nucleus_shapes_key: str | None = "nucleus_boundaries",
    membrane_region: Literal["cell", "nucleus"] = "cell",
    restrict_to_within_boundary: bool = False,
    select_by: Literal["iou", "nucleus_fraction"] = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = 1,
    signed: bool = True,
    inverse_score: bool = True,
    eps: float = 1e-6,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute the mean transcript distance to the boundary ("membrane") of either the cell or
    the matched nucleus.

    For each transcript, the distance is measured to the boundary of the selected polygon:
    - `membrane_region="cell"`: uses `sdata.shapes[shapes_key]`.
    - `membrane_region="nucleus"`: matches each cell to a nucleus (see `select_by`,
      `min_intersection_area`) and uses the boundary of that nucleus for all transcripts
      assigned to the cell.

    Distances can be returned as signed (positive inside/on the polygon, negative outside),
    optionally restricted to transcripts within the boundary, and aggregated per cell
    (mean distance). A normalized version divides by `sqrt(cell_area)` as a length scale.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes : str | list[str] | None, optional
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript distances on.
        If None, all genes are used.
    cell_type_key : str, default="transferred_cell_type"
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    cell_type_query : str | list[str] | None, optional
        If provided, compute the metric only for cells whose `cell_type_key` matches these label(s).
    tables_key : str, default="table"
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_area_key : str, default="cell_area"
        Column in the table with cell area (used for normalization).
    points_gene_key : str, default="feature_name"
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, default="transcripts"
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str | int, default="UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key : str, default="cell_boundaries"
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    nucleus_shapes_key : str | None, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons (required if `membrane_region="nucleus"`).
    membrane_region : {"cell","nucleus"}, default="cell"
        Which boundary to use when computing distances.
    restrict_to_within_boundary : bool, default=False
        If True, keep only transcripts that fall within the cell boundary (uses `covers`,
        so boundary points are included).
    select_by : {"iou","nucleus_fraction"}, default="nucleus_fraction"
        Criterion to choose the best nucleus for each cell when `membrane_region="nucleus"`.
    min_intersection_area : float, default=0.0
        Minimum overlap area required to consider a nucleus a candidate for a cell.
    n_jobs : int, default=1
        Number of parallel jobs for cell-nucleus matching (if needed). `-1` uses all CPUs.
    signed : bool, default=True
        If True, returns signed distances (positive if transcript is inside/on the polygon,
        negative if outside). If False, returns unsigned distances to the boundary.
    inverse_score : bool, default=True
        If True, also computes an inverse-style score that is high when distance is small:
        `1 / sqrt(abs(distance) + eps)`.
    eps : float, default=1e-6
        Small constant for numerical stability in `inverse_score`.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        If `inplace=False`, returns a DataFrame with per-cell mean distance columns:
        - `distance_to_{membrane_region}_membrane_norm_<feature>`
        - optionally `distance_to_{membrane_region}_membrane_inverse_<feature>`
        If `inplace=True`, returns the DataFrame that was merged into `.obs`.
    """
    # validate inputs
    if membrane_region not in ("cell", "nucleus"):
        raise ValueError(f"membrane_region={membrane_region!r} not supported. Use 'cell' or 'nucleus'.")
    if membrane_region == "nucleus" and not nucleus_shapes_key:
        raise ValueError("membrane_region='nucleus' requires `nucleus_shapes_key` to be not None.")

    # transformations alignment check (only for the shapes actually used)
    T_transcripts = sdata.points[points_key].attrs["transform"]

    T_shapes = sdata.shapes[shapes_key].attrs["transform"]
    assert np.array_equal(xy_scale(T_transcripts), xy_scale(T_shapes)), (
        "Cell shapes and transcripts are not aligned. Please ensure they share the same transformation."
    )
    if membrane_region == "nucleus":
        T_shapes = sdata.shapes[nucleus_shapes_key].attrs["transform"]
        assert np.array_equal(xy_scale(T_transcripts), xy_scale(T_shapes)), (
            "Nucleus shapes and transcripts are not aligned. Please ensure they share the same transformation."
        )

    tbl = sdata.tables[tables_key]

    # filter transcripts
    transcript_df = _get_filtered_points_df(
        sdata=sdata,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
    )

    # get selected boundary geometry per cell (cell polygons or matched nucleus polygons)
    _, boundary_gdf = _get_cell_geometry_lookup(
        sdata=sdata,
        region=membrane_region,
        shapes_key=shapes_key,
        nucleus_shapes_key=nucleus_shapes_key,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_x_key=points_x_key,
        points_y_key=points_y_key,
        select_by=select_by,
        min_intersection_area=min_intersection_area,
        n_jobs=n_jobs,
        inplace=inplace,
    )

    boundary_gdf = boundary_gdf[["geometry"]].copy()

    # optionally restrict transcripts to be inside the CELL boundary (independent of membrane_region)
    if restrict_to_within_boundary:
        cell_boundary = sdata.shapes[shapes_key][["geometry"]].copy()
        cell_boundary = gpd.GeoDataFrame(cell_boundary, geometry="geometry").rename(
            columns={"geometry": "cell_geometry"}
        )

        tmp_cell = transcript_df.merge(
            cell_boundary,
            left_on=points_cell_id_key,
            right_index=True,
            how="inner",
        )
        if tmp_cell.empty:
            raise ValueError(
                "No transcripts remained after joining transcripts to cell boundaries. "
                "Check that points_cell_id_key matches the cell boundary index."
            )

        pt_cell = gpd.GeoSeries(
            gpd.points_from_xy(tmp_cell[points_x_key], tmp_cell[points_y_key]), index=tmp_cell.index
        )
        poly_cell = gpd.GeoSeries(tmp_cell["cell_geometry"], index=tmp_cell.index)
        within_cell = poly_cell.covers(pt_cell)

        tmp_cell = tmp_cell.loc[within_cell].copy()
        transcript_df = tmp_cell.loc[:, transcript_df.columns]
        pt_cell = pt_cell.loc[within_cell]  # keep aligned points

        if transcript_df.empty:
            raise ValueError(
                "No transcripts remain after restrict_to_within_boundary=True. "
                "Consider disabling it or verifying boundaries/transforms."
            )

    # Attach geometry to compute distances to membranes
    if membrane_region == "cell" and restrict_to_within_boundary:
        # reuse the already merged cell geometry (avoid extra merge)
        tmp = tmp_cell.rename(columns={"cell_geometry": "geometry"}).copy()
        pt = pt_cell  # reuse points, no recompute
    else:
        tmp = transcript_df.merge(boundary_gdf, left_on=points_cell_id_key, right_index=True, how="inner")
        pt = gpd.GeoSeries(gpd.points_from_xy(tmp[points_x_key], tmp[points_y_key]), index=tmp.index)

        if tmp.empty:
            raise ValueError(
                "No transcripts remained after joining transcripts to selected boundaries. "
                "Check that points_cell_id_key matches the boundary index."
            )

    poly = gpd.GeoSeries(tmp["geometry"], index=tmp.index)
    dist = pt.distance(poly.boundary)

    # signed distance if requested
    if signed:
        is_within = poly.covers(pt)  # includes boundary points
        dist = dist.where(is_within, -dist)

    # decide feature label
    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = genes[0] if len(genes) == 1 else f"{len(genes)}_genes"

    tmp[f"distance_to_{membrane_region}_membrane_{feature}"] = np.asarray(dist)

    # aggregate per cell (mean)
    mean_df = (
        tmp.groupby(points_cell_id_key, sort=False)[[f"distance_to_{membrane_region}_membrane_{feature}"]]
        .mean()
        .reset_index()
    )

    # add cell area and normalize by cell length scale sqrt(cell_area), independent of membrane_region
    area_df = tbl.obs[[tables_cell_id_key, tables_area_key]]
    mean_df = mean_df.merge(area_df, left_on=points_cell_id_key, right_on=tables_cell_id_key, how="left")

    mean_df[f"distance_to_{membrane_region}_membrane_norm_{feature}"] = mean_df[
        f"distance_to_{membrane_region}_membrane_{feature}"
    ] / np.sqrt(mean_df[tables_area_key])

    if inverse_score:
        mean_df[f"distance_to_{membrane_region}_membrane_inverse_{feature}"] = 1.0 / np.sqrt(
            np.abs(mean_df[f"distance_to_{membrane_region}_membrane_{feature}"]) + eps
        )

    mean_df = mean_df.reset_index(drop=True)

    if inplace:
        cols = [tables_cell_id_key, f"distance_to_{membrane_region}_membrane_norm_{feature}"]
        if inverse_score:
            cols.append(f"distance_to_{membrane_region}_membrane_inverse_{feature}")

        out = mean_df[cols].copy()

        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        return out

    return mean_df


def membrane_distance_skewness(
    sdata: sd.SpatialData,
    genes: str | list[str] | None = None,
    cell_type_key: str = "transferred_cell_type",
    cell_type_query: str | list[str] | None = None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    min_transcripts: int = 5,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell skewness of transcript distances to the cell boundary (membrane),
    using only transcripts that are geometrically within/on the cell polygon.

    The function optionally filters by `cell_type_query`, selects non-background transcripts
    assigned to those cells (optionally by `genes`), keeps transcripts inside or on the cell polygon,
    computes their distance to the polygon boundary, and aggregates these distances per cell to obtain
    skewness, returning NaN for mean and skewness when fewer than min_transcripts are available.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    genes : str | list[str] | None, optional
        String or list of strings indicating the feature/gene(s) to calculate the mean transcript distances on.
        If None, all genes are used.
    cell_type_key : str, default="transferred_cell_type"
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    cell_type_query : str | list[str] | None, optional
        If provided, compute the metric only for cells whose `cell_type_key` matches these label(s).
    tables_key : str, default="table"
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_area_key : str, default="cell_area"
        Column in the table with cell area (used for normalization).
    points_gene_key : str, default="feature_name"
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, default="transcripts"
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    points_background_id : str | int, default="UNASSIGNED"
        The cell ID value indicating background transcripts that should be ignored.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key : str, default="cell_boundaries"
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    min_transcripts: int, default=20
        Miinimum number of transcripts required to compute a skewness.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        Per-cell results with columns:
        - points_cell_id_key
        - `skew_dist_to_{membrane_region}_membrane_<feature>`

        where `<feature>` is:
        - `"all_genes"` if `genes is None`
        - the gene name if `genes` is a single string
        - `"<k>_genes"` if `genes` is a list of length k (k>1)

    Raises
    ------
    ValueError
        If no transcripts remain after filtering/joining/within-cell restriction.
    """
    # decide feature label for column naming
    if genes is None:
        feature = "all_genes"
    elif isinstance(genes, str):
        feature = genes
    else:
        feature = genes[0] if len(genes) == 1 else f"{len(genes)}_genes"

    # filter transcripts by cell type + genes + background
    transcript_df = _get_filtered_points_df(
        sdata=sdata,
        genes=genes,
        cell_type_key=cell_type_key,
        cell_type_query=cell_type_query,
        tables_key=tables_key,
        tables_cell_id_key=tables_cell_id_key,
        points_key=points_key,
        points_cell_id_key=points_cell_id_key,
        points_gene_key=points_gene_key,
        points_background_id=points_background_id,
    )

    # load cell boundary polygons
    cells_gdf = sdata.shapes[shapes_key]

    # keep only transcripts whose cell_id exists in shapes index
    transcript_df = transcript_df[transcript_df[points_cell_id_key].isin(cells_gdf.index)].copy()
    if transcript_df.empty:
        raise ValueError(
            "No transcripts remained after filtering to cells present in shapes. "
            "Check that sdata.shapes[shapes_key].index matches transcript cell IDs."
        )

    # join polygon geometry to each transcript
    tmp = transcript_df.join(cells_gdf.geometry.rename("cell_geometry"), on=points_cell_id_key, how="inner")
    if tmp.empty:
        raise ValueError(
            "Transcript-to-cell geometry join produced no rows. Check `points_cell_id_key` and shapes index."
        )

    # build transcript point geometries
    points = gpd.GeoSeries(
        gpd.points_from_xy(tmp[points_x_key].to_numpy(), tmp[points_y_key].to_numpy()),
        index=tmp.index,
    )
    polys = gpd.GeoSeries(tmp["cell_geometry"].to_numpy(), index=tmp.index)

    # restrict to inside/on boundary
    inside = polys.covers(points)
    tmp = tmp.loc[inside].copy()
    points = points.loc[inside]
    polys = polys.loc[inside]
    if tmp.empty:
        raise ValueError("No transcripts remain after restricting to transcripts geometrically within cells. ")

    # Distance to cell boundary (unsigned)
    dist = points.distance(polys.boundary)
    tmp[f"dist_to_cell_membrane_{feature}"] = dist.to_numpy()

    # Aggregate per cell
    grouped = tmp.groupby(points_cell_id_key, sort=False)[f"dist_to_cell_membrane_{feature}"]

    out = pd.DataFrame(
        {
            points_cell_id_key: grouped.mean().index,
            f"skew_dist_to_cell_membrane_{feature}": grouped.apply(
                lambda s: _fisher_pearson_sample_skew(s.to_numpy())
            ).values,
            f"n_transcripts_used_{feature}": grouped.size().values,
        }
    )

    # Mask low-count cells
    low = out[f"n_transcripts_used_{feature}"] < min_transcripts
    out.loc[low, [f"skew_dist_to_cell_membrane_{feature}"]] = np.nan
    out = out[[points_cell_id_key, f"skew_dist_to_cell_membrane_{feature}"]]

    # Merge into obs if requested
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=points_cell_id_key,
        )

    return out
