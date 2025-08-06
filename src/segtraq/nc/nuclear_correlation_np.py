import numpy as np
import spatialdata as sd
import pandas as pd
from tqdm import tqdm

def _compute_iou_cell_raster(cell_id, nuc_ids, cell_arr, nuc_arr):
    """
    Compute IoU between one cell mask and candidate nuclei masks.
    Returns best IoU and matching nucleus ID.
    """
    mask_cell = (cell_arr == cell_id)
    best_iou, best_nuc = 0.0, None
    for nid in nuc_ids:
        mask_nuc = (nuc_arr == nid)
        inter = np.logical_and(mask_cell, mask_nuc).sum()
        union = np.logical_or(mask_cell, mask_nuc).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_nuc = iou, int(nid)
    return {"cell_id": int(cell_id), "best_nuc_id": best_nuc, "IoU": best_iou}


def compute_cell_nuc_ious(
    sdata,
    scale: str = "scale2",
    cell_label_key: str = "cell_labels",
    nuc_label_key: str = "nucleus_labels",
    use_progress: bool = True
):
    """
    Compute per-cell raster-based IoU between mismatched cell/nucleus masks.

    Parameters
    ----------
    sdata : SpatialData
      Contains label arrays at keys `cell_label_key` and `nuc_label_key`.
    scale : str
      On which scale to compute IoUs. 
    cell_label_key : str
      Label image key for cells.
    nuc_label_key : str
      Label image key for nuclei.
    use_progress : bool
      Show progress bar over cells.

    Returns
    -------
    pandas.DataFrame
      Columns: [cell_id, best_nuc_id, IoU]
    """
    cell_arr = sdata.labels[cell_label_key][f"/{scale}/image"].values
    nuc_arr = sdata.labels[nuc_label_key][f"/{scale}/image"].values

    cell_ids = np.unique(cell_arr)
    cell_ids = cell_ids[cell_ids != 0]

    iterator = cell_ids
    if use_progress:
        iterator = tqdm(iterator, desc="Processing raster IoU per cell", total=len(cell_ids))

    results = []
    for cid in iterator:
        overlapping = np.unique(nuc_arr[cell_arr == cid])
        overlapping = overlapping[overlapping != 0]
        if overlapping.size == 0:
            results.append({"cell_id": int(cid), "best_nuc_id": None, "IoU": 0.0})
        else:
            results.append(_compute_iou_cell_raster(cid, overlapping, cell_arr, nuc_arr))

    return pd.DataFrame(results)
