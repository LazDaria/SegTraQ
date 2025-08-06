import geopandas as gpd
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import spatialdata as sd
from pandas import DataFrame, Series
from tqdm import tqdm
from shapely.geometry.base import BaseGeometry
from rtree.index import Index


def _compute_iou(
        poly1: BaseGeometry, 
        poly2: BaseGeometry
) -> float:
    
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
) -> dict[str, int | None | float]:
    
    """For one cell polygon compute the IoU with the best-matching nucleus."""

    cell_geom = cell_row.geometry

    # Get candidate nuclei bounding boxes that overlap this cell's bbox
    candidate_idx = list(nuc_sindex.intersection(cell_geom.bounds))

    if not candidate_idx:
        return {"cell_id": cell_row.name, "best_nuc_id": None, "IoU": 0.0}

    candidates = nuc_boundaries.iloc[candidate_idx]

    best_iou: float = 0.0
    best_nuc_id: int | None = None
    for _, nuc in candidates.iterrows():
        nuc_geom = nuc.geometry
        iou = _compute_iou(cell_geom, nuc_geom)
        if pd.notna(iou) and iou > best_iou:
            best_iou = iou
            best_nuc_id = nuc.name

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
        delayed(_process_cell)(cell_row, nuc_boundaries, nuc_sindex)
        for _, cell_row in iterator
    )

    return pd.DataFrame(results)
