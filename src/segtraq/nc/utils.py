import numpy as np
import warnings
import pandas as pd
import spatialdata as sd
from geopandas import GeoDataFrame
from pandas import Series
from rtree.index import Index
from shapely.geometry.base import BaseGeometry
from spatialdata.models import PointsModel


def _compute_iou(poly1: BaseGeometry, poly2: BaseGeometry) -> float:
    """Compute IoU between two shape polygons."""

    if not (poly1.is_valid and poly2.is_valid):  # TODO - make polygons valid later
        return np.nan
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0


def _process_cell(
    cell_row: Series,
    shapes_cell_id_key: str | None,
    id_key: str | None,
    nuc_boundaries: GeoDataFrame,
    nuc_sindex: Index,
) -> dict[str | int, str | int, int | None | float]:
    """For one cell polygon compute the IoU with the best-matching nucleus."""

    cell_geom = cell_row.geometry

    cell_id = cell_row[shapes_cell_id_key] if shapes_cell_id_key is not None else cell_row.name

    # Get candidate nuclei bounding boxes that overlap this cell's bbox
    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    if not candidate_idx:
        return {id_key: cell_row.name, "best_nuc_id": np.nan, "IoU": 0.0}

    candidates = nuc_boundaries.iloc[candidate_idx]

    best_iou: float = 0.0
    best_nuc_id = np.nan
    for _, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        iou = _compute_iou(cell_geom, nuc_geom)
        if pd.notna(iou) and iou > best_iou:
            best_iou = iou
            best_nuc_id = nuc.name

    return {id_key: cell_id, "best_nuc_id": best_nuc_id, "IoU": best_iou}


def _nucleus_by_feature_df(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    nucleus_shapes_key: str = "nucleus_boundaries",
    points_gene_key: str = "feature_name",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str | None = "z",
) -> pd.DataFrame:
    """
    Aggregate feature counts per nucleus, converting transcripts to 2D if needed.

    Parameters
    ----------
        sdata : SpatialData
            A `SpatialData` object containing segmented and transcript-assigned spatial
            transcriptomics data (images, tables, points, shapes and optional labels).
        nucleus_shapes_key : str, default="nucleus_boundaries"
            Key in `sdata.shapes` for nucleus boundary polygons, if available.
        points_x_key : str, default="x"
            Column for the x-coordinate of each transcript/spot.
        points_gene_key : str, default="feature_name"
            Column specifying the gene/feature name for each transcript/spot.
        points_x_key : str, default="x"
            Column for the x-coordinate of each transcript/spot.
        points_y_key : str, default="y"
            Column for the y-coordinate of each transcript/spot.
        points_z_key : str or None, optional, default="z"
            Column for the z-coordinate (3D data). If `None`, data are treated as 2D.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by nucleus ID, columns = features (genes/proteins), values = counts.
    """

    pts = sdata.points[points_key]
    # check dimensionality: assume 3D if "z" in actual data columns
    df = pts.compute()
    is_3d = points_z_key in df.columns  # TODO - maybe there is a better way to check if transcripts are 3D

    if is_3d:
        transcripts_2d_key = points_key + "_2D"
        df2 = df.drop(columns=[points_z_key])
        coord_sys = "global"  # TODO find an soft coded way to get coordinate system of transcripts
        trans = sd.transformations.get_transformation(pts, to_coordinate_system=coord_sys, get_all=False)

        if hasattr(trans, "scale") and hasattr(trans, "axes"):
            # reduce transformation to 2D to avoid shape mismatch error
            trans.scale = trans.scale[:2]
            trans.axes = trans.axes[:2]

        trans_dict = {coord_sys: trans}

        if not df2.index.is_unique:
            warnings.warn(
                "Index of sdata.points[points_key] is not unique — resetting index to avoid reindexing errors.",
                UserWarning,
            )
            df2 = df2.reset_index(drop=True)

        pts2 = PointsModel.parse(
            df2,
            name=transcripts_2d_key,
            coordinates={"x": points_x_key, "y": points_y_key},
            transformations=trans_dict,
        )
        sdata.points[transcripts_2d_key] = pts2
        value_key = transcripts_2d_key
    else:
        value_key = points_key

    # perform aggregation
    sdata2 = sdata.aggregate(
        values=value_key,
        by=nucleus_shapes_key,
        value_key=points_gene_key,
        agg_func="count",
        deep_copy=False,
    )
    ad = sdata2.tables["table"]
    X = ad.X
    arr = X.toarray() if hasattr(X, "toarray") else X
    df_out = pd.DataFrame(arr, index=sdata2[nucleus_shapes_key].index, columns=ad.var_names)
    return df_out
