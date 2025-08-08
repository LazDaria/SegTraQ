import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from pandas import DataFrame, Series
from rtree.index import Index
from scipy.stats import pearsonr
from shapely.geometry.base import BaseGeometry
from spatialdata.models import PointsModel
from tqdm import tqdm

# from typing import Optional, List
# from scipy.sparse import csr_matrix


def _compute_iou(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    """Compute IoU between two shape polygons."""

    if not (poly1.is_valid and poly2.is_valid):  # TODO - make polygons valid later
        return np.nan
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0


def _process_cell(
    cell_row: Series,
    nuc_boundaries: GeoDataFrame,
    nuc_sindex: Index,
) -> dict[str, int | None | str]:
    """For one cell polygon compute the IoU with the best-matching nucleus."""

    cell_geom = cell_row.geometry

    # Get candidate nuclei bounding boxes that overlap this cell's bbox
    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    if not candidate_idx:
        return {"cell_id": cell_row.name, "best_nuc_id": None, "IoU": 0.0}

    candidates = nuc_boundaries.iloc[candidate_idx]

    best_iou: float = 0.0
    best_nuc_id: str | None = None
    for _, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        iou = _compute_iou(cell_geom, nuc_geom)
        if pd.notna(iou) and iou > best_iou:
            best_iou = iou
            best_nuc_id = str(nuc.name)

    return {"cell_id": cell_row.name, "best_nuc_id": best_nuc_id, "IoU": best_iou}


def compute_cell_nuc_ious(
    sdata: sd.SpatialData,
    cell_shape_key: str = "cell_boundaries",
    nuc_shape_key: str = "nucleus_boundaries",
    n_jobs: int = -1,
    use_progress: bool = True,
) -> DataFrame:
    """
    Compute per-cell IoU between cell and nucleus boundaries in a SpatialData object.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Must contain cell and nuclear shapes.
    cell_shape_key : str, optional
        The key in the `shapes` attribute of `sdata` that corresponds to cell boundaries.
    nuc_shape_key : str, optional
        The key in the `shapes` attribute of `sdata` that corresponds to nucleus boundaries.
    n_jobs : int, optional
        Number of parallel jobs. Default=-1 uses all CPUs.
    use_progress : bool, optional
        Whether to display a progress bar with tqdm.

    Returns
    -------
    pandas.DataFrame
        Columns: [cell_id, best_nuc_id, IoU]
    """

    # Get GeoDataFrames
    cell_boundaries = sdata.shapes[cell_shape_key]
    nuc_boundaries = sdata.shapes[nuc_shape_key]

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

    # Parallel loop over cells
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(_process_cell)(cell_row, nuc_boundaries, nuc_sindex) for _, cell_row in iterator
    )

    return pd.DataFrame(results)


def _nucleus_by_feature_df(
    sdata: sd.SpatialData,
    transcripts_key: str = "transcripts",
    nucleus_by: str = "nucleus_boundaries",
    feature_column: str = "feature_name",
    x_coordinate: str = "x",
    y_coordinate: str = "y",
) -> pd.DataFrame:
    """
    Aggregate feature counts per nucleus, converting transcripts to 2D if needed.

    Parameters
    ----------
    sdata : SpatialData
        `SpatialData` containing transcript `Points` and nucleus `Shapes`.
    transcripts_key : str
        Name of transcripts `Points` element.
    nucleus_by : str
        Name of nucleus shape layer to aggregate by.
    feature_column : str
        Column in transcripts pointing to feature (e.g. gene/protein).
    x_coordinate: str
        Column in transcripts pointing x coordinate.
    y_coordinate: str
        Column in transcripts pointing y coordinate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by nucleus ID, columns = features (genes/proteins), values = counts.
    """

    pts = sdata.points[transcripts_key]
    # check dimensionality: assume 3D if "z" in actual data columns
    df = pts.compute()
    is_3d = "z" in df.columns  # TODO - maybe there is a better way to check if transcripts are 3D

    if is_3d:
        transcripts_2d_key = transcripts_key + "_2D"
        df2 = df.drop(columns=["z"])
        coord_sys = "global"  # TODO find an soft coded way to get coordinate system of transcripts
        trans = sd.transformations.get_transformation(pts, to_coordinate_system=coord_sys, get_all=False)

        if hasattr(trans, "scale") and hasattr(trans, "axes"):
            # reduce transformation to 2D to avoid shape mismatch error
            trans.scale = trans.scale[:2]
            trans.axes = trans.axes[:2]

        trans_dict = {coord_sys: trans}

        pts2 = PointsModel.parse(
            df2,
            name=transcripts_2d_key,
            coordinates={"x": x_coordinate, "y": y_coordinate},
            transformations=trans_dict,
        )
        sdata.points[transcripts_2d_key] = pts2
        value_key = transcripts_2d_key
    else:
        value_key = transcripts_key

    # perform aggregation
    sdata2 = sdata.aggregate(
        values=value_key,
        by=nucleus_by,
        value_key=feature_column,
        agg_func="count",
        deep_copy=False,
    )
    ad = sdata2.tables["table"]
    X = ad.X
    arr = X.toarray() if hasattr(X, "toarray") else X
    df_out = pd.DataFrame(arr, index=sdata2["nucleus_boundaries"].index.astype(str), columns=ad.var_names)
    return df_out


def compute_cell_nuc_correlation(
    sdata: sd.SpatialData,
    table_key: str = "table",
    cell_id_key: str = "cell_id",
    metric: str = "pearson",
    transcripts_key: str = "transcripts",
    nucleus_by: str = "nucleus_boundaries",
    feature_column: str = "feature_name",
    x_coordinate: str = "x",
    y_coordinate: str = "y",
) -> pd.DataFrame:
    """
    For each cell in the SpatialData table, identifies the nucleus with highest IoU
    and computes a correlation (e.g. Pearson) between the gene expression profiles
    of the cell and that nucleus.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A SpatialData object containing:
          - `.shapes['cell_boundaries']` and `.shapes['nucleus_boundaries']`
            for polygon geometries,
          - `.tables[table_key]` as an AnnData table.
    table_key : str
        Key in `sdata.tables` pointing to the expression matrix.
    cell_id_key : str
        Column in `sdata.tables[table_key].obs containing cell IDs to match with shapes.
    metric : str
        Correlation metric. Currently supports only `"pearson"`.
    transcripts_key : str
        Name of transcripts `Points` element.
    nucleus_by : str
        Name of nucleus shape layer to aggregate by.
    feature_column : str
        Column in transcripts pointing to feature (e.g. gene/protein).
    x_coordinate: str
        Column in transcripts pointing x coordinate.
    y_coordinate: str
        Column in transcripts pointing y coordinate.


    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
          - `cell_id`: identifier of each cell,
          - `best_nuc_id`: matching nucleus ID with highest IoU (or None),
          - `correlation`: Pearson correlation between the cell and its matched nucleus gene counts
            (NaN if no match).
    """

    df = sdata.tables[table_key].obs.copy()
    if "best_nuc_id" not in df.columns:
        iou_df = compute_cell_nuc_ious(sdata)
        df = df.merge(
            iou_df.set_index("cell_id"),
            left_on=cell_id_key,
            right_index=True,
            how="left",
        )

    arr = (
        sdata.tables[table_key].X.toarray()
        if hasattr(sdata.tables[table_key].X, "toarray")
        else sdata.tables[table_key].X
    )
    expr_cells = pd.DataFrame(
        arr,
        index=sdata.tables[table_key].obs[cell_id_key],
        columns=sdata.tables[table_key].var.index,
    )

    expr_nucleus_df = _nucleus_by_feature_df(
        sdata, transcripts_key, nucleus_by, feature_column, x_coordinate, y_coordinate
    )
    expr_nucleus = expr_nucleus_df[expr_cells.columns]

    rows = []
    for _, row in df.iterrows():
        cid, nid = row.cell_id, row.best_nuc_id
        if pd.isna(nid):  # if no overlapping nucleus
            rows.append(
                {
                    "cell_id": cid,
                    "best_nuc_id": None,
                    "IoU": row.IoU,
                    "correlation": 0.0,
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
                    "cell_id": cid,
                    "best_nuc_id": nid,
                    "IoU": row.IoU,
                    "correlation": corr,
                }
            )
    return pd.DataFrame(rows)


def compute_correlation_between_parts(
    sdata: sd.SpatialData,
    table_key: str = "table",
    cell_shape_key: str = "cell_boundaries",
    nuc_shape_key: str = "nucleus_boundaries",
    transcripts_key: str = "transcripts",
    feature_column: str = "feature_name",
    x_coordinate: str = "x",
    y_coordinate: str = "y",
) -> pd.DataFrame:
    """
    Compute Pearson correlation between cell part overlapping with its nucleus and the rest of the cell.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing cells, nuclei, and transcript points.
    table_key : str
        Key in `sdata.tables` pointing to the expression matrix.
    cell_shape_key : str
        Key for cell boundaries in sdata.shapes.
    nuc_shape_key : str
        Key for nucleus boundaries in sdata.shapes.
    transcripts_key : str
        Key for transcript points in sdata.points.
    feature_column : str
        Feature column in transcript points (e.g. gene name).
    x_coordinate : str
        Column name for x coordinate.
    y_coordinate : str
        Column name for y coordinate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["cell_id", "best_nuc_id", "correlation"]
    """

    # Step 1: Compute IoUs if needed
    if "best_nuc_id" not in sdata.tables[table_key].obs.columns:
        iou_df = compute_cell_nuc_ious(sdata, cell_shape_key, nuc_shape_key)
    else:
        iou_df = sdata.tables[table_key].obs[["cell_id", "best_nuc_id", "IoU"]].copy()

    # Set up geometries
    cell_gdf = sdata.shapes[cell_shape_key]
    nuc_gdf = sdata.shapes[nuc_shape_key]
    transcripts_df = sdata.points[transcripts_key].compute()

    # Assuming transcripts_df has 'x' and 'y' columns
    transcripts_gdf = gpd.GeoDataFrame(
        transcripts_df,
        geometry=gpd.points_from_xy(transcripts_df[x_coordinate], transcripts_df[y_coordinate]),
        crs=nuc_gdf.crs,  # TODO - how do make sure that they are actually in the same coordinate system
    )

    results = []
    for row in tqdm(iou_df.itertuples(index=False), desc="Computing correlations"):
        cell_id, best_nuc_id, iou_val = row.cell_id, row.best_nuc_id, row.IoU

        if best_nuc_id is None:
            results.append(
                {
                    "cell_id": cell_id,
                    "best_nuc_id": best_nuc_id,
                    "IoU": iou_val,
                    "correlation_parts": np.nan, 
                }
            )  # TODO np.nan
            continue

        cell_geom = cell_gdf.loc[cell_id].geometry
        nuc_geom = nuc_gdf.loc[
            int(best_nuc_id)
        ].geometry  # TODO - sdata['table'].obs_names is str, but shapes are int - check how they link

        if not (cell_geom.is_valid and nuc_geom.is_valid):
            results.append(
                {
                    "cell_id": cell_id,
                    "best_nuc_id": best_nuc_id,
                    "IoU": iou_val,
                    "correlation_parts": np.nan,
                }
            )  # TODO np.nan
            continue

        intersection = cell_geom.intersection(nuc_geom)
        remainder = cell_geom.difference(intersection)

        if intersection.is_empty or remainder.is_empty:
            results.append(
                {
                    "cell_id": cell_id,
                    "best_nuc_id": best_nuc_id,
                    "IoU": iou_val,
                    "correlation_parts": np.nan,
                }
            )  # TODO np.nan
            continue

        # Select transcripts inside each part
        pts_intersection = transcripts_gdf[transcripts_gdf.geometry.within(intersection)]
        pts_remainder = transcripts_gdf[transcripts_gdf.geometry.within(remainder)]

        # Count features
        count_intersection = pts_intersection[feature_column].value_counts()
        count_remainder = pts_remainder[feature_column].value_counts()

        # Restrict to var_names from sdata.tables[table_key]
        valid_features = sdata.tables[table_key].var_names
        filtered_features = count_intersection.index.union(count_remainder.index).intersection(valid_features)

        # Create aligned vectors for Pearson correlation
        vec1 = count_intersection.reindex(filtered_features, fill_value=0)
        vec2 = count_remainder.reindex(filtered_features, fill_value=0)

        if vec1.sum() == 0 or vec2.sum() == 0:
            corr = np.nan
        else:
            corr, _ = pearsonr(vec1, vec2)

        results.append(
            {
                "cell_id": cell_id,
                "best_nuc_id": best_nuc_id,
                "IoU": iou_val,
                "correlation_parts": corr,
            }
        )

    return pd.DataFrame(results)
