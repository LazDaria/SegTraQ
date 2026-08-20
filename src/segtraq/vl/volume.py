from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import spatialdata as sd
from joblib import Parallel, delayed
from ovrlpy import Ovrlp, cell_integrity_from_transcripts
from shapely.ops import unary_union

from .._settings import settings
from ..rs.utils import _two_profile_similarity_metrics
from ..utils import _ensure_index, _get_genes, _is_background, merge_into_obs
from .utils import _correct_z_drift, _run_ovrlpy


def vertical_signal_integrity_per_cell(
    sdata,
    ovrlp: Ovrlp | None = None,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    n_jobs: int | None = None,
    random_state: int = 123,
    ovrlpy_init_kwargs: dict[str, Any] | None = None,
    ovrlpy_analyse_kwargs: dict[str, Any] | None = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell mean VSI by sampling a precomputed VSI map at transcript locations.

    This metric assumes `vsi_map` is defined on the same coordinate system and scaling
    as the transcript coordinates in `sdata.points[points_key]`, i.e. VSI values are
    indexed directly by integer x/y coordinates after optional shift-to-origin.

    If `ovrlp` is provided, the existing `ovrlpy.Ovrlp` object is used directly. If
    `ovrlp=None`, ovrlpy is run internally. For each transcript, the
    VSI value is read from the ovrlpy object and then averaged across transcripts
    belonging to each cell.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object.
    ovrlp : ovrlpy.Ovrlp or None, default=None
        Precomputed ovrlpy object with VSI already calculated. If `None`, ovrlpy is run
        internally using `n_comp`.
    points_key : str, default="transcripts"
        Key in `sdata.points` for the transcript-level points table.
    points_gene_key : str, default="feature_name"
        Column in the points table containing gene names.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_x_key : str, default="x"
        Column in the points table containing transcript x-coordinates.
    points_y_key : str, default="y"
        Column in the points table containing transcript y-coordinates.
    points_z_key : str, default="z"
        Column in the points table containing transcript z-coordinates.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table. If `inplace=True`,
        results are merged into `sdata.tables[tables_key].obs`.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    n_jobs : int  or None, default=None
        Number of jobs passed to `ovrlpy.Ovrlp` if `ovrlp=None`.
    random_state : int, default=42
        Random seed passed to `ovrlpy.Ovrlp` if `ovrlp=None`.
    ovrlpy_init_kwargs : dict or None, default=None
        Additional keyword arguments passed to `ovrlpy.Ovrlp` if `ovrlp=None`.
    ovrlpy_analyse_kwargs : dict or None, default=None
        Additional keyword arguments passed to `ovrlpy.Ovrlp.analyse` if `ovrlp=None`.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [tables_cell_id_key, "vertical_signal_integrity"].
    """
    if n_jobs is None:
        n_jobs = settings.n_jobs

    if ovrlp is None:
        ovrlp = _run_ovrlpy(
            sdata=sdata,
            n_jobs=n_jobs,
            points_key=points_key,
            points_gene_key=points_gene_key,
            points_cell_id_key=points_cell_id_key,
            points_x_key=points_x_key,
            points_y_key=points_y_key,
            points_z_key=points_z_key,
            random_state=random_state,
            ovrlpy_init_kwargs=ovrlpy_init_kwargs,
            ovrlpy_analyse_kwargs=ovrlpy_analyse_kwargs,
        )

    vsi = cell_integrity_from_transcripts(
        ovrlp,
        cell_id=points_cell_id_key,
        unassigned=points_background_id,
    )

    vsi_per_cell = (
        vsi.group_by(points_cell_id_key)
        .agg(pl.col("vsi").mean())
        .rename({"vsi": "vertical_signal_integrity"})
        .to_pandas()
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=vsi_per_cell,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=points_cell_id_key,
        )

    return vsi_per_cell


def similarity_top_bottom(
    sdata,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int = "UNASSIGNED",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str = "z",
    correct_z_drift: bool = True,
    max_points: int = 1_000_000,
    seed: int | None = 0,
    q: float = 0.30,
    scale: float = 1e4,
    min_genes: int = 5,
    min_transcripts: int = 10,
    n_permutations: int = 200,
    random_state: int | None = 42,
    n_jobs: int | None = None,
    parallel_backend: str = "threading",
    inplace: bool = True,
) -> pd.DataFrame:
    """Compute PFlog1pPF cosine similarity between the bottom and top of each cell.

    Transcripts are split into bottom and top regions using within-cell z-quantiles,
    optionally after correcting global z-drift. The resulting count profiles are
    transformed with PFlog1pPF (shifted CLR) and compared using cosine similarity;
    larger values indicate more similar transcript composition across cell depth.

    If `n_permutations >= 100`, a conditional permutation test pools the bottom and
    top counts and reallocates transcripts while preserving both region totals.
    The lower-tail p-value quantifies whether the observed similarity is smaller
    than expected under a shared-composition null model.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing transcript-assigned spatial transcriptomics data.
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    points_key : str, default="transcripts"
        Key in `sdata.points` for transcript-level data.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript to a cell.
    points_background_id : str or int, default="UNASSIGNED"
        Identifier for transcripts not assigned to any cell (background).
    points_gene_key : str, default="feature_name"
        Column specifying the gene/feature name for each transcript.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript.
    points_z_key : str, default="z"
        Column specifying the z coordinate / depth for each transcript.
    correct_z_drift : bool, default=True
        If True, correct global z-drift before computing within-cell z-quantiles.
        The corrected values are used only for defining top/bottom subsets.
    max_points : int, default=1_000_000
        Maximum number of points used to fit the regression for z-drift correction.
    seed : int or None, default=0
        Random seed used for subsampling during z-drift correction. If `None`,
        sampling is not reproducible.
    q : float, default=0.30
        Quantile defining the bottom and top parts: bottom <= q and top >= 1 - q.
    scale : float, default=1e4
        Scale factor used in the PFlog1pPF transformation.
    min_genes : int, default=5
        Minimum number of genes with nonzero counts across bottom and top required
        to score a cell.
    min_transcripts : int, default=10
        Minimum number of transcripts required in each part to score a cell.
    n_permutations : int, default=200
        Number of conditional permutations used to compute the p-value. Must be >= 100.
    random_state : int or None, default=42
        Random seed used to generate reproducible per-cell permutation streams.
    n_jobs : int  or None, default=None
        Number of parallel jobs used for per-cell profile comparisons.
    parallel_backend : str, default="threading"
        Parallelization backend passed to joblib.
    inplace : bool, default=True
        Whether to add the results to `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with `tables_cell_id_key` and `similarity_top_bottom`. If
        `n_permutations >= 100`, `similarity_top_bottom_p_value` is also returned.
    """
    if n_jobs is None:
        n_jobs = settings.n_jobs

    if not (0.0 < q < 0.5):
        raise ValueError(f"`q` must be in (0, 0.5). Got {q}.")
    if n_permutations < 100:
        raise ValueError("`n_permutations` must be >= 100.")

    # Subset points and drop rows with missing cell identifiers, genes, or coordinates.
    pts = sdata.points[points_key]
    cols = [points_cell_id_key, points_gene_key, points_x_key, points_y_key, points_z_key]
    tx = pts[cols].dropna(subset=cols)

    # Remove background transcripts and genes that are absent from the cell table.
    is_bg = _is_background(tx[points_cell_id_key], points_background_id)
    tx = tx[~is_bg]
    all_genes = _get_genes(
        adata=sdata.tables[tables_key],
        gene_key=tables_gene_key,
    )
    tx = tx[tx[points_gene_key].isin(all_genes)]

    tx = tx.compute() if hasattr(tx, "compute") else tx
    tx = tx.reset_index(drop=True)

    # Corrected z values are used only to define the top/bottom subsets.
    if correct_z_drift:
        tx["_z_for_split"] = _correct_z_drift(
            tx=tx,
            points_x_key=points_x_key,
            points_y_key=points_y_key,
            points_z_key=points_z_key,
            max_points=max_points,
            seed=seed,
        )
    else:
        tx["_z_for_split"] = tx[points_z_key].to_numpy(dtype=float)

    # Compute within-cell quantile cutoffs and exclude cells without a valid z range.
    grouped_z = tx.groupby(points_cell_id_key, observed=True)["_z_for_split"]
    tx["_z_bottom"] = grouped_z.transform(lambda s: s.quantile(q))
    tx["_z_top"] = grouped_z.transform(lambda s: s.quantile(1.0 - q))
    valid_split = tx["_z_bottom"] < tx["_z_top"]
    tx["_is_bottom"] = valid_split & (tx["_z_for_split"] <= tx["_z_bottom"])
    tx["_is_top"] = valid_split & (tx["_z_for_split"] >= tx["_z_top"])

    counts_bottom = (
        tx[tx["_is_bottom"]].groupby([points_cell_id_key, points_gene_key], observed=True).size().unstack(fill_value=0)
    )
    counts_top = (
        tx[tx["_is_top"]].groupby([points_cell_id_key, points_gene_key], observed=True).size().unstack(fill_value=0)
    )

    all_cells = pd.Index(sdata.tables[tables_key].obs[tables_cell_id_key])
    counts_bottom = counts_bottom.reindex(index=all_cells, columns=all_genes, fill_value=0)
    counts_top = counts_top.reindex(index=all_cells, columns=all_genes, fill_value=0)

    # Materialize dense count matrices once before entering the parallel loop.
    counts_bottom_values = counts_bottom.to_numpy(dtype=int, copy=False)
    counts_top_values = counts_top.to_numpy(dtype=int, copy=False)

    # Seeds are needed only when a permutation p-value is requested.
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(all_cells), dtype=np.uint32)
    else:
        seeds = [None] * len(all_cells)

    def _score_cell(i, cid, cell_seed):
        metrics = _two_profile_similarity_metrics(
            counts_bottom_values[i],
            counts_top_values[i],
            n_permutations=n_permutations,
            min_transcripts=min_transcripts,
            min_genes=min_genes,
            scale=scale,
            rng=(np.random.default_rng(int(cell_seed)) if cell_seed is not None else None),
        )
        row = {
            tables_cell_id_key: cid,
            "similarity_top_bottom": metrics["similarity"],
        }
        if "similarity_p_value" in metrics:
            row["similarity_top_bottom_p_value"] = metrics["similarity_p_value"]
        return row

    rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(_score_cell)(i, cid, cell_seed)
        for i, (cid, cell_seed) in enumerate(zip(all_cells, seeds, strict=False))
    )
    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError(
            f"Could not compute top-bottom profile similarities. Try a different quantile. You used q={q}."
        )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return out


def fraction_heterotypic_overlap(
    sdata: sd.SpatialData,
    shapes_key_list: list[str] | None,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    cell_type_key: str = "transferred_cell_type",
    shapes_cell_id_key: str = "cell_id",
    unknown_label: str = "Unknown",
    unknown_policy: str = "treat_as_label",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute cross-depth heterotypic overlap fraction per cell using one representative polygon per cell
    (chosen as the polygon with the largest area across z layers).

    For a representative polygon i (cell_id, z_layer) with geometry P_i and type t_i:

        overlap_area_i = Area( P_i ∩ Union_{j: z_j != z_i, id_j != id_i, t_j != t_i} P_j )
        overlap_fraction_i = overlap_area_i / Area(P_i)

    Candidates are restricted to bbox-overlapping polygons via a spatial index.

    Unknown/NA types:
      - unknown_policy="exclude": cells with NA/unknown type return NaN, and unknown-type
        candidates are excluded from overlap.
      - unknown_policy="treat_as_label": NA is replaced by `unknown_label` and treated as a
        real category.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing cell boundary polygons in multiple z layers and a
        cell table with transferred cell type labels.
    shapes_key_list : list[str]
        Keys in `sdata.shapes` for per-z-layer cell boundary polygons
        (e.g. ["cell_boundaries_z0", ..., "cell_boundaries_z3"]).
    tables_key : str, default="table"
        Key in `sdata.tables` for the cell-level metadata table.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    cell_type_key : str, default="transferred_cell_type"
        Column in the cell table containing cell-type labels (e.g. transferred from scRNA-seq).
    shapes_cell_id_key : str, optional, default="cell_id"
        Index name of shapes GeoDataFrame linking polygons to cell IDs.
    unknown_label : str, default="Unknown"
        Label name to use when treating NA as a separate category (unknown_policy="treat_as_label").
    unknown_policy : str, default="exclude"
        How to handle Unknown/NA cell types:
          - "exclude": exclude polygons with NA/unknown types from comparisons. If the focal cell
            has NA/unknown type, its overlap fraction is set to NaN.
          - "treat_as_label": convert NA to `unknown_label` and treat it as a valid category.
    inplace : bool, default=True
        Whether to merge the aggregated per-cell result into `sdata.tables[tables_key].obs`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [tables_cell_id_key, "heterotypic_overlap_area", "heterotypic_overlap_fraction"].
    """
    if unknown_policy not in {"exclude", "treat_as_label"}:
        raise ValueError(f"unknown_policy must be one of {{'exclude','treat_as_label'}}. Got: {unknown_policy}")

    # extract meta data of cell observations
    obs = sdata.tables[tables_key].obs
    if cell_type_key not in obs.columns:
        raise KeyError(f"{cell_type_key!r} not found in sdata.tables[{tables_key!r}].obs")

    cell_type_map = obs.set_index(tables_cell_id_key)[cell_type_key].copy()

    gdfs = []
    # loop over the $z$-stacks
    for k, skey in enumerate(shapes_key_list):
        if skey not in sdata.shapes:
            raise KeyError(f"shapes key {skey!r} not found in sdata.shapes")
        # extract geopandas dataframe
        shapes = sdata.shapes[skey].copy()

        gdf = _ensure_index(shapes, shapes_key=skey, id_key=shapes_cell_id_key, id_key_name="shapes_cell_id_key")

        gdf["_cell_id"] = gdf.index
        # assign $z$-layer
        gdf["_z_layer"] = k
        gdf["_cell_type"] = gdf["_cell_id"].map(cell_type_map)

        if unknown_policy == "treat_as_label":
            gdf["_cell_type"] = gdf["_cell_type"].astype("object").where(gdf["_cell_type"].notna(), other=unknown_label)
        # subset geometry to be not NA and not empty
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        gdfs.append(gdf[["_cell_id", "_z_layer", "_cell_type", "geometry"]])

    gdf_all = pd.concat(gdfs, ignore_index=True)
    # making sure the concatenated table is a proper GeoPandas GeoDataFrame
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry="geometry", crs=gdfs[0].crs if len(gdfs) else None)

    if unknown_policy == "exclude":
        gdf_all["_is_unknown"] = gdf_all["_cell_type"].isna() | (
            gdf_all["_cell_type"].astype("object") == unknown_label
        )
    else:
        gdf_all["_is_unknown"] = False

    gdf_all["_area"] = gdf_all.geometry.area.replace(0, np.nan)

    # pick representative polygon per cell: max area across layers
    rep_idx = gdf_all.groupby("_cell_id")["_area"].idxmax()
    reps = gdf_all.loc[rep_idx].copy()
    # get spatial index for cells in bbox around gdf polygons
    sindex = gdf_all.sindex

    # compute overlap fraction only for representative polygons
    out_rows = []
    for i, row in reps.iterrows():
        cid = row["_cell_id"]
        z_i = row["_z_layer"]
        t_i = row["_cell_type"]
        geom_i = row.geometry
        area_i = row["_area"]
        # store invalid cell areas
        if area_i is None or np.isnan(area_i) or area_i <= 0:
            out_rows.append((cid, np.nan, np.nan))
            continue

        if unknown_policy == "exclude":
            if bool(row["_is_unknown"]) or pd.isna(t_i):
                out_rows.append((cid, np.nan, np.nan))
                continue
        # get indices of the intersections of the spatial index with the
        # target cell
        cand_idx = list(sindex.intersection(geom_i.bounds))
        # exclude itself
        cand_idx = [j for j in cand_idx if j != i]
        # if empty add 0.0 - do here to avoid empty indexing
        if not cand_idx:
            out_rows.append((cid, 0.0, 0.0))
            continue

        cands = gdf_all.iloc[cand_idx]
        # only consider candidates that are not in the same $z$-layer
        cands = cands[cands["_z_layer"] != z_i]
        # only consider cells that are not the same cell_id (across $z$)
        cands = cands[cands["_cell_id"] != cid]

        if unknown_policy == "exclude":
            cands = cands[~cands["_is_unknown"]]
        # only consider cell types that are not of the same cell type
        cands = cands[cands["_cell_type"] != t_i]
        # if empty add 0.0
        if cands.empty:
            out_rows.append((cid, 0.0, 0.0))
            continue
        # compute area intersections between candidate cells and
        # target cell
        inter_geoms = []
        for geom_j in cands.geometry:
            inter = geom_i.intersection(geom_j)
            if (not inter.is_empty) and (inter.area > 0):
                inter_geoms.append(inter)

        if not inter_geoms:
            overlap_area = 0.0
        else:
            overlap_area = float(unary_union(inter_geoms).area)

        out_rows.append((cid, overlap_area, overlap_area / float(area_i)))

    per_cell_overlap_fraction = pd.DataFrame(
        out_rows, columns=[tables_cell_id_key, "heterotypic_overlap_area", "heterotypic_overlap_fraction"]
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_overlap_fraction,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return per_cell_overlap_fraction
