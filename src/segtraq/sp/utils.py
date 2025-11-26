import numpy as np
import pandas as pd
import squidpy as sq
from sklearn.metrics import average_precision_score


def add_neighbor_celltype_binary(
    adata,
    cell_type_col="transferred_cell_type",
    tables_x_key: str = "x",
    tables_y_key: str = "y",
):
    adata.obsm["spatial"] = adata.obs[[tables_x_key, tables_y_key]].to_numpy()

    # 1. Build spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    G = adata.obsp["spatial_connectivities"]

    # 2. Prepare clean cell-type array (no NaNs, pure strings)
    if cell_type_col is None or cell_type_col not in adata.obs:
        raise ValueError(
            f"Cell type column '{cell_type_col}' not found in adata.obs. \n"
            f"Please provide a valid column name with the cell_type_col parameter. \n"
            f"Available columns: {list(adata.obs.columns)}"
        )
    cell_types = adata.obs[cell_type_col].astype("category").cat.add_categories("Unknown").fillna("Unknown")
    ct_categories = cell_types.cat.categories
    ct_index = {ct: i for i, ct in enumerate(ct_categories)}

    # 3. Allocate result matrix
    out = np.zeros((adata.n_obs, len(ct_categories)), dtype=np.uint8)

    # 4. Neighbor index pointer arrays
    indptr = G.indptr
    indices = G.indices
    ct_array = cell_types.values

    # 5. Fill binary neighbor matrix
    for i in range(adata.n_obs):
        neighbors = indices[indptr[i] : indptr[i + 1]]
        if len(neighbors) == 0:
            continue
        neigh_cts = np.unique(ct_array[neighbors])
        for ct in neigh_cts:
            out[i, ct_index[ct]] = 1

    # 6. Store as DataFrame for nice labeling
    adata.obsm["neighbor_celltype_binary"] = pd.DataFrame(out, index=adata.obs_names, columns=ct_categories)

    return adata


def assign_grid_splits(
    adata, mask_cells, grid_shape=(10, 10), test_size=0.25, seed=0, tables_x_key="x", tables_y_key="y"
):
    """
    Assign train/test splits based on spatial grid units to prevent spatial leakage.

    This function divides the spatial coordinate space into a grid. It assigns
    whole grid tiles to either the training or test set, ensuring that cells
    physically close to each other (within the same tile) are not split across sets.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    mask_cells : np.ndarray or list
        Boolean mask or indices indicating the subset of cells to split (e.g., focal cell type).
    grid_shape : tuple of int, optional
        The number of grid units in (x, y) directions. Default is (10, 10).
    test_size : float, optional
        The proportion of grid units to assign to the test set. Default is 0.25.
    seed : int, optional
        Random seed for reproducible grid shuffling. Default is 0.
    tables_x_key : str, optional
        Column name in adata.obs for X coordinates. Default is "x".
    tables_y_key : str, optional
        Column name in adata.obs for Y coordinates. Default is "y".

    Returns
    -------
    is_train : np.ndarray
        Boolean mask corresponding to `mask_cells` (subset length) indicating training samples.
    is_test : np.ndarray
        Boolean mask corresponding to `mask_cells` (subset length) indicating test samples.
    """
    rng = np.random.RandomState(seed)

    # Extract coordinates for the specific subset of cells
    xs = adata.obs[tables_x_key].values[mask_cells].astype(float)
    ys = adata.obs[tables_y_key].values[mask_cells].astype(float)

    # Normalize spatial coordinates to [0,1] for grid assignment
    def _norm(v):
        mn, mx = np.nanmin(v), np.nanmax(v)
        return np.zeros_like(v) if mx == mn else (v - mn) / (mx - mn)

    xsn, ysn = _norm(xs), _norm(ys)

    # Assign cells to grid IDs
    gx = np.minimum((xsn * grid_shape[0]).astype(int), grid_shape[0] - 1)
    gy = np.minimum((ysn * grid_shape[1]).astype(int), grid_shape[1] - 1)
    grid_ids = gx + gy * grid_shape[0]

    # Split the unique grids, not the cells directly
    unique_grids = np.unique(grid_ids)
    rng.shuffle(unique_grids)

    n_total = len(unique_grids)
    n_test = int(np.floor(n_total * test_size))
    n_train = n_total - n_test  # Ensure all grids are used

    train_grids = unique_grids[:n_train]
    test_grids = unique_grids[n_train:]

    # Map back to cell-level masks
    is_train = np.isin(grid_ids, train_grids)
    is_test = np.isin(grid_ids, test_grids)

    return is_train, is_test


def test_model_above_chance(y_true, y_pred, n_bootstrap=1000, seed=0):
    """
    Test whether the model's average precision is above chance using bootstrap.

    Parameters
    ----------
    y_true : array-like (n_samples,)
        True binary labels.
    y_pred : array-like (n_samples,)
        Model prediction scores (logits or probabilities).
    n_bootstrap : int
        Number of bootstrap samples to generate null distribution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    p_value : float
        Empirical p-value.
    null_aps : array
        Array of bootstrap APs under the null.
    observed_ap : float
        Average precision on the observed data.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    observed_ap = average_precision_score(y_true, y_pred)
    n = len(y_true)

    null_aps = []
    for _ in range(n_bootstrap):
        # simulate null: generate random scores with same positive fraction
        y_null_scores = rng.uniform(size=n)
        # or could shuffle y_pred: y_null_scores = rng.permutation(y_pred)
        # labels unchanged (or use y_true as reference)
        null_ap = average_precision_score(y_true, y_null_scores)
        null_aps.append(null_ap)

    null_aps = np.array(null_aps)
    p_value = np.mean(null_aps >= observed_ap)

    return p_value, null_aps, observed_ap


def run_single_permutation(
    actual_probs,
    y_test,
    seed,
):
    """Calculates score for a permuted null distribution using fixed predictions."""
    # Use a local RNG for thread safety
    local_rng = np.random.RandomState(seed)

    # 1. Permute test labels (y_test)
    y_test_permuted = local_rng.permutation(y_test)

    # 2. Score: The ACTUAL predictions are scored against the PERMUTED labels
    #    This tests the null hypothesis that the *true* association between features (in X_test)
    #    and labels (in y_test) is broken.
    return average_precision_score(y_test_permuted, actual_probs)
