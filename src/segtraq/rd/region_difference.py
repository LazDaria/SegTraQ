import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from pandas import DataFrame

from ..utils import _get_count_matrix, _get_genes, merge_into_obs
from .utils import (
    _border_admixture_permutation_metrics,
    _get_center_border_counts,
    _get_neighborhood_counts,
    _join_points_regions,
    _match_nucleus_one_cell,
    _two_profile_permutation_metrics,
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
    """Compute the best-matching nucleus for each cell.

    Matching is based on Intersection-over-Union (IoU) or nucleus fraction
    (area(cell ∩ nucleus) / area(nucleus)). Candidate nuclei are identified by
    spatial overlap, and the highest-scoring nucleus is retained for each cell.
    The final result also enforces a one-to-one assignment so that a nucleus is
    not matched to multiple cells. 

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
    n_jobs : int, default=-1
        Number of parallel jobs across cells. `-1` uses all available CPU cores.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        One row per cell with the matched nucleus ID, IoU, and nucleus fraction.
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

    # Reuse the nucleus spatial index across cells to avoid repeated index construction.
    cell_boundaries = sdata.shapes[shapes_key]
    nuc_boundaries = sdata.shapes[nucleus_shapes_key]
    nuc_sindex = nuc_boundaries.sindex

    # Each cell can be matched independently. Threading avoids repeatedly copying
    # the GeoDataFrames and spatial index to worker processes.
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

    # Resolve nuclei matched to multiple cells by keeping only the best cell-nucleus
    # match according to the primary overlap score, using the secondary score as a tie-breaker.
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



def _rename_difference_metrics(metrics: dict, prefix: str) -> dict:
    """Attach a public metric prefix to shared two-profile outputs."""
    return {
        f"{prefix}_difference": metrics["difference"],
        f"{prefix}_difference_p_value": metrics["difference_p_value"],
    }


def _cell_seeds(n: int, random_state: int | None) -> np.ndarray:
    """Generate deterministic, independent seeds for cell-wise permutations."""
    rng = np.random.default_rng(random_state)
    return rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)


def nucleus_cell_difference(
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
    tables_raw_counts_layer: str | None = None,
    min_transcripts: int = 10,
    min_genes: int = 5,
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    inplace: bool = True,
    n_permutations: int = 200,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compare whole-cell and matched-nucleus profiles using a G-statistic difference score.

    The whole-cell expression profile is compared with the transcript profile
    inside its matched nucleus. The returned difference score is a
    bias-corrected, count-normalized G statistic, with larger values indicating
    stronger differences in gene composition. Its permutation p-value is computed
    from a conditional null that preserves pooled gene counts and both region totals.

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
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.
    n_permutations : int, default=200
        Number of conditional permutations per cell.
    random_state : int or None, default=42
        Seed for reproducible cell-wise permutations.

    Returns
    -------
    pd.DataFrame
        One row per cell with nucleus-match information, the difference
        score, and its permutation p-value.
    """

    assert nucleus_shapes_key is not None, (
        "Cannot compute nucleus-cell difference: `nucleus_shapes_key` is None. "
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
    count_genes = pd.Index(_get_genes(adata=adata, gene_key=tables_gene_key))
    cell_positions = pd.Series(
        np.arange(adata.n_obs),
        index=adata.obs[tables_cell_id_key],
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

    # Keep the cell count matrix sparse and materialize only the genes used in the
    # nucleus comparison for one cell at a time.
    common_genes = expr_nucleus.columns.intersection(count_genes)
    gene_positions = count_genes.get_indexer(common_genes)
    expr_nucleus = expr_nucleus[common_genes]
    seeds = _cell_seeds(len(match_df), random_state)

    def _cell_count_vector(cid) -> np.ndarray:
        row = counts[cell_positions.loc[cid], gene_positions]
        if hasattr(row, "toarray"):
            row = row.toarray()
        return np.asarray(row).ravel()

    def _compute_one(row: pd.Series, seed: np.uint32) -> dict:
        cid, nid = row[shapes_cell_id_key], row["nucleus_id"]
        if pd.isna(nid):
            base_metrics = {
                key: np.nan
                for key in (
                    "difference",
                    "difference_p_value",
                )
            }
        else:
            base_metrics = _two_profile_permutation_metrics(
                _cell_count_vector(cid),
                expr_nucleus.loc[nid].to_numpy(),
                n_permutations=n_permutations,
                min_transcripts=min_transcripts,
                min_genes=min_genes,
                rng=np.random.default_rng(int(seed)),
            )

        return {
            tables_cell_id_key: cid,
            "nucleus_id": nid,
            "iou": row.iou,
            "nucleus_fraction": row.nucleus_fraction,
            **_rename_difference_metrics(base_metrics, "nucleus_cell_difference"),
        }

    # Permutations are independent across cells, so parallelize at the cell level.
    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_compute_one)(row, seed)
        for (_, row), seed in zip(match_df.iterrows(), seeds, strict=False)
    )

    out = pd.DataFrame(rows)
    if inplace:
        merge_into_obs(
            sdata=sdata, 
            tables_key=tables_key, 
            df_to_merge=out, 
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key)
    return out


def nucleus_cytoplasm_difference(
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
    select_by: str = "nucleus_fraction",
    min_intersection_area: float = 0.0,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    inplace: bool = True,
    n_permutations: int = 200,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compare matched nuclear and cytoplasmic profiles using a G-statistic difference score.

    For each cell, transcripts are separated into those inside the matched
    nucleus and those in the remaining cytoplasmic region. Their expression
    profiles are then compared against a conditional permutation null to
    quantify whether the two compartments differ more than expected by chance.
    This specifically tests intracellular compartmentalization of transcripts while
    controlling for the different numbers of transcripts in nucleus and cytoplasm.

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
    select_by : str, default="nucleus_fraction"
        Score used to select the best-matching nucleus per cell. Options:
        - "iou": maximize Intersection-over-Union (cell vs nucleus).
        - "nucleus_fraction": maximize area(cell ∩ nucleus) / area(nucleus).
        If multiple nuclei have the same score, the larger nucleus is selected.
    min_intersection_area : float, default=0.0
        Minimum area (cell ∩ nucleus) required to consider a nucleus as a
        candidate. Overlaps <= this threshold are ignored.
    n_jobs : int, default=-1
        Number of parallel jobs across cells. `-1` uses all available CPU cores.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.
    n_permutations : int, default=200
        Number of conditional permutations per cell.
    random_state : int or None, default=42
        Seed for reproducible cell-wise permutations.

    Returns
    -------
    pd.DataFrame
        One row per cell with nucleus-match information and null-calibrated
        nucleus-cytoplasm comparison metrics.
    """

    assert nucleus_shapes_key is not None, (
        "Cannot compute nucleus-cytoplasm difference: `nucleus_shapes_key` is None. "
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

    seeds = _cell_seeds(len(all_cells), random_state)

    def _compute_one(cid, seed: np.uint32) -> dict:
        if pd.isna(best_nuc_map.get(cid)):
            base_metrics = {
                key: np.nan
                for key in (
                    "difference",
                    "difference_p_value",
                )
            }
        else:
            base_metrics = _two_profile_permutation_metrics(
                counts_intersection.loc[cid].to_numpy(dtype=int),
                counts_cytoplasm.loc[cid].to_numpy(dtype=int),
                n_permutations=n_permutations,
                min_transcripts=min_transcripts,
                min_genes=min_genes,
                rng=np.random.default_rng(int(seed)),
            )
        return {
            id_key: cid,
            **_rename_difference_metrics(base_metrics, "nucleus_cytoplasm_difference"),
        }

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_compute_one)(cid, seed)
        for cid, seed in zip(all_cells, seeds, strict=False)
    )

    sim_df = pd.DataFrame(rows)
    out = match_df.reset_index(drop=True).merge(sim_df, on=id_key, how="left")
    if inplace:
        merge_into_obs(
            sdata=sdata, 
            tables_key=tables_key, 
            df_to_merge=out, 
            tables_cell_id_key=tables_cell_id_key, 
            df_cell_id_key=id_key)
    return out


def center_border_difference(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    min_transcripts: int = 10,
    min_genes: int = 5,
    inplace: bool = True,
    n_permutations: int = 200,
    random_state: int | None = 42,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
) -> pd.DataFrame:
    """Compare center and border profiles using a G-statistic difference score.

    Each cell is partitioned into an inner center and an outer border, separated
    by an optional buffer region. The corresponding transcript profiles are
    compared against a conditional permutation null to quantify within-cell
    expression differences while accounting for finite transcript counts. The
    buffer prevents transcripts close to the center-border boundary from contributing
    to either profile, reducing sensitivity to the exact geometric split.

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
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.
    n_permutations : int, default=200
        Number of conditional permutations per cell.
    random_state : int or None, default=42
        Seed for reproducible cell-wise permutations.
    n_jobs : int, default=-1
        Number of parallel jobs across cells. `-1` uses all available CPU cores.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.

    Returns
    -------
    pd.DataFrame
        One row per cell with difference score and permutation p-value.
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

    seeds = _cell_seeds(len(all_cells), random_state)

    def _compute_one(cid, seed: np.uint32) -> dict:
        metrics = _two_profile_permutation_metrics(
            expr_center.loc[cid].to_numpy(dtype=int),
            expr_border.loc[cid].to_numpy(dtype=int),
            n_permutations=n_permutations,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            rng=np.random.default_rng(int(seed)),
        )
        return {id_key: cid, **_rename_difference_metrics(metrics, "center_border_difference")}

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_compute_one)(cid, seed)
        for cid, seed in zip(all_cells, seeds, strict=False)
    )
    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Could not compute center-border profile comparisons. " f"You used {border_fraction_of_radius=}.")
    
    if inplace:
        merge_into_obs(
            sdata=sdata, 
            tables_key=tables_key, 
            df_to_merge=out, 
            tables_cell_id_key=tables_cell_id_key, 
            df_cell_id_key=id_key)
        
    return out


def border_neighborhood_difference(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    neighborhood_radius_factor: float = 1.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    inplace: bool = True,
    n_permutations: int = 200,
    random_state: int | None = 42,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
) -> pd.DataFrame:
    """Compare border and neighborhood profiles using a G-statistic difference score.

    For each cell, the expression profile of its border is compared with the
    aggregate profile of nearby cells defined by the neighborhood radius. A
    conditional permutation null is used to distinguish systematic similarity
    or differences from effects expected from finite transcript sampling. This
    comparison can reveal whether transcripts near a cell boundary resemble the
    surrounding cellular environment rather than the interior of the focal cell.

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
    inplace : bool, default=True
        Whether to merge the results into `sdata.tables[tables_key].obs`.
    n_permutations : int, default=200
        Number of conditional permutations per cell.
    random_state : int or None, default=42
        Seed for reproducible cell-wise permutations.
    n_jobs : int, default=-1
        Number of parallel jobs across cells. `-1` uses all available CPU cores.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.

    Returns
    -------
    pd.DataFrame
        One row per cell with difference score and permutation p-value.
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

    seeds = _cell_seeds(len(all_cells), random_state)

    def _compute_one(cid, seed: np.uint32) -> dict:
        metrics = _two_profile_permutation_metrics(
            expr_border.loc[cid].to_numpy(dtype=int),
            expr_neighborhood.loc[cid].to_numpy(dtype=int),
            n_permutations=n_permutations,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            rng=np.random.default_rng(int(seed)),
        )
        return {
            id_key: cid,
            **_rename_difference_metrics(metrics, "border_neighborhood_difference"),
        }

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_compute_one)(cid, seed)
        for cid, seed in zip(all_cells, seeds, strict=False)
    )
    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            "Could not compute border-neighborhood profile comparisons. "
            f"You used {border_fraction_of_radius=} and {neighborhood_radius_factor=}."
        )
    
    if inplace:
        merge_into_obs(
            sdata=sdata, 
            tables_key=tables_key, 
            df_to_merge=out, 
            tables_cell_id_key=tables_cell_id_key, 
            df_cell_id_key=id_key)
        
    return out


def border_admixture_score(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    shapes_key: str = "cell_boundaries",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_gene_key: str = "feature_name",
    border_fraction_of_radius: float = 0.2,
    buffer_fraction_of_radius: float = 0.1,
    neighborhood_radius_factor: float = 1.0,
    min_transcripts: int = 10,
    min_genes: int = 5,
    pseudocount: float = 0.5,
    n_permutations: int = 200,
    random_state: int | None = 42,
    n_jobs: int = -1,
    parallel_backend: str = "threading",
    inplace: bool = True,
) -> pd.DataFrame:
    """Return null-corrected border admixture improvement and permutation p-value.

    The metric asks whether a cell's border profile is better explained as a
    mixture of its center and neighboring-cell expression than by its center
    alone. The mixture coefficient is fitted between the center and neighborhood
    profiles, and the resulting improvement quantifies how much adding the neighborhood
    component improves the border fit. The observed improvement is calibrated against
    a permutation null, with bootstrap confidence intervals used to summarize uncertainty.

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
    points_background_id : str or int, default="UNASSIGNED"
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
        Neighbor distance threshold expressed as a multiple of the median
        equivalent cell radius.
    min_transcripts : int, default=10
        Minimum number of transcripts required in each region.
    min_genes : int, default=5
        Minimum number of genes required across the three regions combined.
    pseudocount : float, default=0.5
        Pseudocount used when converting counts to proportions.
    n_boot : int, default=200
        Number of permutations per cell. Retained under the existing parameter
        name for API compatibility.
    random_state : int or None, default=42
        Seed for reproducible cell-wise permutations.
    n_jobs : int, default=-1
        Number of parallel jobs across cells. `-1` uses all available CPU cores.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.
    inplace : bool, default=True
        If True, merge the results into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        One row per cell with null-corrected border-admixture score and
        permutation p-value.
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
    
    seeds = _cell_seeds(len(common_cells), random_state)

    def _one_cell(cid, seed):
        result = _border_admixture_permutation_metrics(
            x_center=expr_center.loc[cid].to_numpy(dtype=int),
            x_border=expr_border.loc[cid].to_numpy(dtype=int),
            x_neighborhood=expr_neighborhood.loc[cid].to_numpy(dtype=int),
            n_permutations=n_permutations, min_transcripts=min_transcripts,
            min_genes=min_genes, pseudocount=pseudocount,
            rng=np.random.default_rng(int(seed)),
        )
        return {id_key: cid, **result}

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_one_cell)(cid, seed) for cid, seed in zip(common_cells, seeds, strict=False)
    )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "Could not compute border admixture scores. "
            f"You used {border_fraction_of_radius=} and {neighborhood_radius_factor=}."
        )

    if inplace:
        merge_into_obs(
            sdata=sdata, 
            tables_key=tables_key, 
            df_to_merge=out, 
            tables_cell_id_key=tables_cell_id_key, 
            df_cell_id_key=id_key)
            
    return out
