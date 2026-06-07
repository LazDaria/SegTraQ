import collections
import copy
import math
import warnings
from collections.abc import Callable
from importlib.metadata import version

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from dask import delayed
from joblib import Parallel
from joblib import delayed as joblib_delayed
from packaging import version as pkg_version
from rasterio.features import shapes
from scipy import sparse
from scipy.spatial.distance import cdist
from shapely.affinity import affine_transform, translate
from shapely.geometry import shape
from sklearn.metrics import roc_auc_score
from spatialdata.models import PointsModel
from spatialdata.transformations import (
    get_transformation,
    get_transformation_between_coordinate_systems,
    set_transformation,
)

from .bl import baseline as bl
from .constants import CONNECTIVITIES_KEY, DISTANCES_KEY, NEIGHBORS_KEY, NORM_LOG_LAYER, PCA_KEY, SEGTRAQ_CELL_ID_KEY


def xy_scale(T):  # TODO - extract Translation, Scale, Sequence
    if hasattr(T, "scale"):
        return np.asarray(T.scale)[:2]
    return np.array([1.0, 1.0])


def _to_ndarray(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _looks_like_counts(x, n: int = 1000, tol: float = 1e-8) -> bool:
    """Quickly check if data looks like non-negative integer counts."""
    if sparse.issparse(x):
        # Nonzero entries only (zeros are fine for count check)
        arr = x.data
    elif hasattr(x, "values") and not isinstance(x, np.ndarray):
        arr = np.asarray(x.values).ravel()
    else:
        arr = np.asarray(x).ravel()

    if arr.size == 0:
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True

    samp = arr if arr.size <= n else np.random.choice(arr, n, replace=False)
    return np.all(samp >= 0) and np.allclose(samp, np.round(samp), atol=tol)


def _get_pca_and_neighbors(
    adata: AnnData,
    raw_layer: str | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    target_sum: float | None = 1e4,
) -> AnnData:
    """
    Compute (or reuse) PCA and neighbors using the pipeline's norm_log layer.

    All results are stored under namespaced keys so they can be
    distinguished from any externally-computed PCA/neighbors:
    - adata.layers[NORM_LOG_LAYER]
    - adata.obsm[PCA_KEY]
    - adata.uns[NEIGHBORS_KEY]
    - adata.obsp[CONNECTIVITIES_KEY], adata.obsp[DISTANCES_KEY]

    Parameters
    ----------
    adata : AnnData
    raw_layer : str or None
        Layer with raw counts. None → use `.X`.
    n_neighbors: int
        Number of neighbors for `sc.pp.neighbors`.
    n_pcs: int
        Number of PCs for `sc.pp.pca` and `sc.pp.neighbors`.
    target_sum: float or None
        If not None, passed as `target_sum` to `sc.pp.normalize_total` when
        computing the norm_log layer. Ignored if the norm_log layer already exists.

    Returns
    -------
    AnnData
        The same object (modified in place), returned for convenience.
    """
    # Step 1: ensure norm_log layer exists
    adata = _get_norm_log(adata, layer=raw_layer, target_sum=target_sum)

    # Step 2: PCA on norm_log if not already done by this pipeline
    if PCA_KEY not in adata.obsm:
        tmp = AnnData(X=adata.layers[NORM_LOG_LAYER].copy())
        sc.pp.pca(tmp, n_comps=n_pcs)
        adata.obsm[PCA_KEY] = tmp.obsm["X_pca"]

    # Step 3: neighbor graph on pipeline PCA if not already done
    if NEIGHBORS_KEY not in adata.uns:
        tmp = AnnData(X=adata.layers[NORM_LOG_LAYER].copy())
        tmp.obsm["X_pca"] = adata.obsm[PCA_KEY]
        sc.pp.neighbors(tmp, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # Store under namespaced keys
        adata.uns[NEIGHBORS_KEY] = tmp.uns["neighbors"]
        adata.obsp[CONNECTIVITIES_KEY] = tmp.obsp["connectivities"]
        adata.obsp[DISTANCES_KEY] = tmp.obsp["distances"]

    return adata


def _apply_overlap_filter(marker_dict: dict[str, list[str]], t, n_ct) -> dict[str, list[str]]:
    all_genes = [g for gl in marker_dict.values() for g in gl]
    if not all_genes:
        return {k: [] for k in marker_dict}
    counts = pd.Series(all_genes).value_counts()
    # drop genes appearing in >= t * n_types lists
    drop_genes = set(counts[counts >= (t * n_ct)].index)

    return {ct: [g for g in gl if g not in drop_genes] for ct, gl in marker_dict.items()}


def _resolve_obs_index_ambiguity(
    sdata: sd.SpatialData,
    tables_key: str,
    tables_cell_id_key: str,
) -> None:
    """
    Ensure tables_cell_id_key exists as a plain column in adata.obs.

    Three cases:
    - Cell ID is only a column → no action needed.
    - Cell ID is only the index name → add it as a column from the index values.
    - Cell ID is both index name and column → verify they are identical and have
      the same dtype, raise if not.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object to fix in place.
    tables_key : str
        Key for accessing the table in sdata.tables.
    tables_cell_id_key : str
        Column name that should exist in obs (and may also be the index name).
    """
    table = sdata.tables[tables_key]
    obs = table.obs

    is_column = tables_cell_id_key in obs.columns
    is_index_name = obs.index.name == tables_cell_id_key

    if is_column and not is_index_name:
        # Perfect — nothing to do
        return

    elif is_index_name and not is_column:
        # Add the index as a column
        obs[tables_cell_id_key] = obs.index.values

    elif is_index_name and is_column:
        # Both exist — verify they are consistent
        index_vals = obs.index.astype(obs[tables_cell_id_key].dtype)
        if not (index_vals == obs[tables_cell_id_key].values).all():
            raise ValueError(
                f"'{tables_cell_id_key}' exists as both the obs index name and a column "
                f"in sdata.tables['{tables_key}'], but their values are not identical. "
                "Please resolve this inconsistency before proceeding."
            )
        if obs.index.dtype != obs[tables_cell_id_key].dtype:
            raise ValueError(
                f"'{tables_cell_id_key}' exists as both the obs index name and a column "
                f"in sdata.tables['{tables_key}'], but their dtypes differ: "
                f"index is {obs.index.dtype}, column is {obs[tables_cell_id_key].dtype}. "
                "Please ensure they have the same dtype before proceeding."
            )
        # Values and dtypes match — no action needed
        return


def _assign_celltype_by_pearson(
    adata: AnnData,
    ref_mean_df: pd.DataFrame,
    tables_gene_key: str | None = None,
    tables_cell_id_key: str = "cell_id",
    genes_to_use: set[str] | None = None,
) -> pd.DataFrame:
    """
    Assign cell types to query cells by Pearson correlation to reference mean profiles.

    Parameters
    ----------
    adata : AnnData
        Query dataset after normalization and log1p transformation.
    ref_mean_df : pandas.DataFrame
        Reference mean expression profiles with cell types as rows and genes as columns.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    tables_cell_id_key : str, default="cell_id"
        Column in `adata.obs` containing unique cell identifiers.
    genes_to_use : set of str or None, default=None
        Optional subset of genes to use for Pearson correlation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns `tables_cell_id_key`, `"transferred_cell_type"`,
        and `"pearson_score"`.
    """
    if tables_cell_id_key not in adata.obs.columns:
        raise KeyError(f"'{tables_cell_id_key}' not found in `adata.obs`.")

    genes = _get_genes(
        adata=adata,
        gene_key=tables_gene_key,
    )

    X_query = pd.DataFrame(
        _to_ndarray(adata.X),
        index=adata.obs[tables_cell_id_key].values,
        columns=genes,
    )

    common_genes = X_query.columns.intersection(ref_mean_df.columns)

    if genes_to_use is not None:
        common_genes = common_genes.intersection(pd.Index(list(genes_to_use)))

    if len(common_genes) == 0:
        raise ValueError("No common genes found between query and reference after filtering.")

    X_query = X_query[common_genes]
    X_ref = ref_mean_df[common_genes]

    cor_mat = 1.0 - cdist(X_query.values, X_ref.values, metric="correlation")
    cor_df = pd.DataFrame(cor_mat, index=X_query.index, columns=X_ref.index)

    best_celltype = cor_df.idxmax(axis=1)
    best_score = cor_df.max(axis=1)

    return pd.DataFrame(
        {
            tables_cell_id_key: X_query.index,
            "transferred_cell_type": best_celltype.values,
            "pearson_score": best_score.values,
        }
    )


def _get_count_matrix(adata, layer: str | None = None, layer_arg: str | None = None):
    """Return raw count matrix from `adata.layers[layer]` or `adata.X`.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing count data.
    layer : str or None, default=None
        Layer containing raw counts. If `None`, counts are expected in `adata.X`.
    layer_arg : str or None, default=None
        Name of the parameter used to specify the layer (for error messages). If `None`, defaults to "raw_layer".

    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        Raw count matrix.
    """
    layer_arg = "raw_layer" if layer_arg is None else layer_arg
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in `adata.layers`.")
        X = adata.layers[layer]
        source = f"adata.layers[{layer!r}]"
    else:
        X = adata.X
        source = "adata.X"

    if not _looks_like_counts(X):
        raise ValueError(
            f"Expected raw count data in `{source}`, but the selected matrix "
            "does not look like non-negative integer counts. "
            f"You can set the layer containing raw counts with the `{layer_arg}` parameter "
            f"(available layers: {list(adata.layers.keys())})."
        )

    return X


def _get_norm_log(
    adata: AnnData,
    layer: str | None = None,
    target_sum: float = 1e4,
    layer_arg: str | None = None,
) -> str:
    """
    Ensure `adata.layers[NORM_LOG_LAYER]` exists and return its key.

    If the layer already exists it is returned immediately (no recomputation).
    Otherwise, raw counts are taken from `layer` (or `.X` if None),
    normalized with `sc.pp.normalize_total`, log-transformed with
    `sc.pp.log1p`, and stored in `adata.layers[NORM_LOG_LAYER]`.

    Parameters
    ----------
    adata : AnnData
    layer : str or None
        Source of raw counts. None → use `.X`.
    target_sum : float
        Passed to `sc.pp.normalize_total`.
    layer_arg : str or None
        Name of the parameter used to specify the layer (for error messages). If `None`, defaults to "raw_layer".

    Returns
    -------
    str
        The key of the normalized+log layer (`NORM_LOG_LAYER`).
    """
    if NORM_LOG_LAYER in adata.layers:
        return adata

    if adata.is_view:
        adata = adata.copy()

    raw = _get_count_matrix(adata, layer=layer, layer_arg=layer_arg)  # validates integer counts

    # Work on a temporary AnnData so sc.pp.* don't touch .X in place
    tmp = AnnData(X=raw.copy())
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)

    adata.layers[NORM_LOG_LAYER] = tmp.X
    return adata


def _get_genes(
    adata: AnnData,
    gene_key: str | None = None,
) -> pd.Index:
    """
    Return gene identifiers from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object`.
    gene_key : str or None, default=None
        Column in `adata.var` containing gene identifiers. If `None`,
        `adata.var_names` are used.

    Returns
    -------
    pandas.Index
        Gene identifiers from `adata.var_names` or `adata.var[gene_key]`.
    """
    if gene_key is None:
        genes = pd.Index(adata.var_names)
    else:
        if gene_key not in adata.var.columns:
            raise KeyError(f"'{gene_key}' not found in `adata.var`.")
        genes = pd.Index(adata.var[gene_key].values)

    if genes.duplicated().any():
        raise ValueError("Gene identifiers are not unique.")

    return genes


def _make_ref_genes_unique(
    adata_ref: AnnData,
    ref_gene_key: str | None = None,
) -> AnnData:
    """
    Ensure gene identifiers used as var_names are unique.

    If `ref_gene_key` is provided, use `adata_ref.var[ref_gene_key]`
    as `var_names`. Duplicate identifiers are made unique using
    `var_names_make_unique()`.
    """
    adata_ref = adata_ref.copy()

    if ref_gene_key is not None:
        if ref_gene_key not in adata_ref.var.columns:
            raise KeyError(f"'{ref_gene_key}' not found in `adata_ref.var`.")

        adata_ref.var_names = adata_ref.var[ref_gene_key].astype(str)

    if not adata_ref.var_names.is_unique:
        warnings.warn(
            "Gene identifiers are not unique. Making them unique with `adata_ref.var_names_make_unique()`.",
            UserWarning,
            stacklevel=2,
        )
        adata_ref.var_names_make_unique()

    return adata_ref


def run_label_transfer(
    sdata,
    adata_ref: AnnData,
    ref_cell_type: str,
    tables_raw_counts_layer: str | None = None,
    ref_raw_counts_layer: str | None = "raw",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    tx_min: float = 10.0,
    tx_max: float = 2000.0,
    gn_min: float = 5.0,
    gn_max: float = np.inf,
    cell_type_key: str = "transferred_cell_type",
    ref_gene_key: str | None = None,
    use_hvg: bool = False,
    exclude_gene_prefixes: tuple[str, ...] = ("MT-", "RPL", "RPS"),
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Transfer cell type labels from a reference AnnData object to cells in a
    SpatialData table using Pearson correlation to reference mean expression profiles.

    Raw counts are selected first, normalized with `sc.pp.normalize_total`,
    log-transformed with `sc.pp.log1p`, and then used for label transfer. If a
    raw-count layer is provided, it is used preferentially. Otherwise, `.X` is
    expected to contain raw counts.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing the query dataset. Cell-level expression
        data are expected in `sdata.tables[tables_key]`.
    adata_ref : AnnData
        Reference AnnData object containing annotated cells.
    ref_cell_type : str
        Column in `adata_ref.obs` containing the reference cell type labels.
    tables_raw_counts_layer : str or None, default=None
        Layer in `sdata.tables[tables_key].layers` containing raw counts for
        the query data. If `None`, raw counts are expected in
        `sdata.tables[tables_key].X`.
    ref_raw_counts_layer : str or None, default=None
        Layer in `adata_ref.layers` containing raw counts for the reference
        data. If `None`, raw counts are expected in `adata_ref.X`.
    tables_key : str, default="table"
        Key identifying the cell-level AnnData table in `sdata.tables`.
    tables_cell_id_key : str, default="cell_id"
        Column in `sdata.tables[tables_key].obs` containing unique cell identifiers.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    points_key : str, default="transcripts"
        Key identifying the transcript-level points element in `sdata.points`.
    points_cell_id_key : str, default="cell_id"
        Column in the transcript points table containing cell identifiers.
    points_gene_key : str, default="feature_name"
        Column in the transcript points table containing gene names.
    tx_min : float, default=10.0
        Minimum number of detected transcripts required for a cell to be retained.
    tx_max : float, default=2000.0
        Maximum number of detected transcripts allowed for a cell to be retained.
    gn_min : float, default=5.0
        Minimum number of detected genes required for a cell to be retained.
    gn_max : float, default=np.inf
        Maximum number of detected genes allowed for a cell to be retained.
    cell_type_key : str, default="transferred_cell_type"
        Column name used to store transferred labels in the query table's `.obs`.
    ref_gene_key : str or None, default=None
        Column in `adata_ref.var` containing gene identifiers.
        If `None`, `adata_ref.var_names` are used.
    use_hvg : bool, default=False
        If `True`, restrict label transfer to highly variable genes computed
        from the reference dataset.
    exclude_gene_prefixes : tuple of str, default=("MT-", "RPL", "RPS")
        Gene prefixes to exclude from the HVG set before label transfer. Set to
        an empty tuple to disable this filtering.
    inplace : bool, default=True
        If `True`, write transferred labels to
        `sdata.tables[tables_key].obs[cell_type_key]` and return `None`.
        If `False`, return a DataFrame with transferred labels and Pearson
        correlation scores.

    Returns
    -------
    pandas.DataFrame or None
        If `inplace=False`, returns a DataFrame with columns including
        `tables_cell_id_key`, `cell_type_key`, and `"pearson_score"`.
        If `inplace=True`, modifies `sdata` in place and returns `None`.
    """
    # copies gene identifiers into var_names and makes them unique (if needed)
    adata_ref = _make_ref_genes_unique(adata_ref, ref_gene_key=ref_gene_key)

    if ref_cell_type not in adata_ref.obs.columns:
        raise KeyError(f"'{ref_cell_type}' not found in `adata_ref.obs`.")

    adata_ref = adata_ref.copy()

    tbl = sdata.tables[tables_key]

    need_tx = "transcript_count" not in tbl.obs.columns
    need_gn = "gene_count" not in tbl.obs.columns

    if need_tx or need_gn:
        bl.transcripts_per_cell(
            sdata,
            tables_cell_id_key=tables_cell_id_key,
            points_key=points_key,
            points_cell_id_key=points_cell_id_key,
            tables_key=tables_key,
        )
        bl.genes_per_cell(
            sdata,
            tables_cell_id_key=tables_cell_id_key,
            points_key=points_key,
            points_cell_id_key=points_cell_id_key,
            points_gene_key=points_gene_key,
            tables_key=tables_key,
        )

    qc_range = {
        "transcript_count": (tx_min, tx_max),
        "gene_count": (gn_min, gn_max),
    }

    mask = np.ones(tbl.n_obs, dtype=bool)

    for key, (low, high) in qc_range.items():
        if key not in tbl.obs.columns:
            raise KeyError(f"QC column '{key}' not found in table `.obs`.")

        values = tbl.obs[key].to_numpy()
        mask &= (values >= low) & (values <= high)

    adata_q = tbl[mask].copy()

    # getting the normalized and log-transformed data into adata_ref and adata_q,
    # stored in a namespaced layer to avoid conflicts
    adata_ref = _get_norm_log(adata_ref, layer=ref_raw_counts_layer, layer_arg="ref_raw_counts_layer")
    adata_q = _get_norm_log(adata_q, layer=tables_raw_counts_layer, layer_arg="tables_raw_counts_layer")

    genes = adata_ref.var_names

    norm_log_counts = _to_ndarray(adata_ref.layers[NORM_LOG_LAYER])
    celltypes = adata_ref.obs[ref_cell_type]

    norm_log_counts_df = pd.DataFrame(norm_log_counts, columns=genes)
    norm_log_counts_df["celltype"] = celltypes.values
    ref_mean_df = norm_log_counts_df.groupby("celltype").mean()

    genes_to_use = None

    if use_hvg:
        sc.pp.highly_variable_genes(
            adata_ref,
            flavor="seurat",
            n_top_genes=2000,
            layer=NORM_LOG_LAYER,
            inplace=True,
        )

        ref_hvg_mask = adata_ref.var["highly_variable"].to_numpy()
        hvgs = set(genes[ref_hvg_mask])

        if exclude_gene_prefixes:
            hvgs = {
                g
                for g in hvgs
                if not any(str(g).upper().startswith(prefix.upper()) for prefix in exclude_gene_prefixes)
            }

        genes_to_use = hvgs

    # using the normalized and log-transformed data for label transfer
    adata_q.X = adata_q.layers[NORM_LOG_LAYER]
    ct_corr = _assign_celltype_by_pearson(
        adata=adata_q,
        ref_mean_df=ref_mean_df,
        tables_gene_key=tables_gene_key,
        tables_cell_id_key=tables_cell_id_key,
        genes_to_use=genes_to_use,
    )

    out = ct_corr.rename(columns={"transferred_cell_type": cell_type_key})

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=out,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        tbl.obs[cell_type_key] = tbl.obs[cell_type_key].astype("category")
        return None

    return out


def merge_into_obs(
    sdata, tables_key, df_to_merge: pd.DataFrame, tables_cell_id_key: str, df_cell_id_key: str, fillna_cols=None
):
    """
    Left-join df_to_merge into sdata.tables[tables_key].obs without resetting the index
    and without creating duplicate key columns.
    - Preserves obs index
    - Uses obs[tables_cell_id_key] as the join key
    - Drops overlapping columns on the right before joining
    """
    obs = sdata.tables[tables_key].obs

    # Temporarily clear the index name to avoid pandas ambiguity when
    # tables_cell_id_key is both a column and the index name
    original_index_name = obs.index.name
    obs.index.name = None

    # Build right indexed by the join key
    right = df_to_merge.set_index(df_cell_id_key, drop=True)

    # Drop overlapping columns from obs to avoid duplicates
    overlapping_cols = [c for c in right.columns if c in obs.columns]
    if overlapping_cols:
        obs = obs.drop(columns=overlapping_cols)

    joined = obs.join(right, on=tables_cell_id_key, how="left")

    # Restore the original index name
    joined.index.name = original_index_name

    # Fill NAs if requested
    if fillna_cols:
        for c in fillna_cols:
            if c in joined.columns:
                joined[c] = joined[c].fillna(0)

    sdata.tables[tables_key].obs = joined


def merge_into_var(sdata, tables_key, df_to_merge):
    var = sdata.tables[tables_key].var

    overlapping = [c for c in df_to_merge.columns if c in var.columns]

    if overlapping:
        var = var.drop(columns=overlapping)

    df = var.merge(df_to_merge, left_index=True, right_index=True, how="left")

    sdata.tables[tables_key].var = df


def _pairwise_auc(
    adata: AnnData,
    gene_key: str | None,
    ctypes: pd.Categorical,
    ref_cell_type: str,
    ct_a: str,
    ct_b: str,
    max_fpr: float | None,
    auc_pos_thresh: float,
    min_cells_per_celltype: int,
) -> tuple[str, str, list[str], bool]:
    """
    Helper: compute per-gene AUC/pAUC for one pair (ct_a, ct_b)
    and return genes up in ct_a vs ct_b.
    """
    # Restrict to cells of ct_a and ct_b
    mask = ctypes.isin([ct_a, ct_b])
    if mask.sum() < 2 * min_cells_per_celltype:
        # too few cells total -> skip
        return (ct_a, ct_b, [], False)

    ad_pair = adata[mask]
    X_pair = ad_pair.X
    if hasattr(X_pair, "toarray"):
        X_pair = X_pair.toarray()
    else:
        X_pair = np.asarray(X_pair)  # (n_cells_pair, n_genes)

    genes = _get_genes(ad_pair, gene_key)
    genes = np.asarray(genes)

    labels = (ad_pair.obs[ref_cell_type].values == ct_a).astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        # Only one class present -> skip
        return (ct_a, ct_b, [], False)

    # Precompute means per group to enforce directionality
    mask_a = labels == 1
    mask_b = labels == 0
    mean_a = X_pair[mask_a].mean(axis=0)
    mean_b = X_pair[mask_b].mean(axis=0)

    # Compute AUC/pAUC per gene (non-vectorized, one roc_auc_score per gene)
    aucs = np.zeros(genes.size, dtype=float)
    for j in range(genes.size):
        scores = X_pair[:, j]
        # If all scores identical, AUC is undefined -> treat as 0.5
        if np.all(scores == scores[0]):
            aucs[j] = 0.5
        else:
            aucs[j] = roc_auc_score(labels, scores, max_fpr=max_fpr)

    # Identify genes up in ct_a vs ct_b
    #   high AUC and higher mean in ct_a
    up_mask = (aucs >= auc_pos_thresh) & (mean_a > mean_b)

    # get indices of candidate genes
    idx = np.where(up_mask)[0]

    # sort candidates by AUC descending
    idx = idx[np.argsort(-aucs[idx])]

    # cap at 200 genes
    idx = idx[:200]

    pos_genes_a = genes[idx].tolist()

    return (ct_a, ct_b, pos_genes_a, True)


def _pairwise_de(
    adata: AnnData,
    ctypes: pd.Categorical,
    ref_cell_type: str,
    ct_a: str,
    ct_b: str,
    method: str,
    pval_adj_thresh: float,
    logfc_pos_thresh: float,
    min_cells_per_celltype: int,
) -> tuple[str, str, list[str], bool]:
    """
    Helper: run DE for one pair (ct_a, ct_b) and return genes up in ct_a.
    """
    mask = ctypes.isin([ct_a, ct_b])
    if mask.sum() < 2 * min_cells_per_celltype:
        # too few cells total, skip
        return (ct_a, ct_b, [], False)

    ad_pair = adata[mask].copy()

    # DE: ct_a vs ct_b
    sc.tl.rank_genes_groups(
        ad_pair,
        groupby=ref_cell_type,
        groups=[ct_a],
        reference=ct_b,
        method=method,
    )
    df = sc.get.rank_genes_groups_df(ad_pair, group=ct_a)

    pos_df = df[(df["pvals_adj"] < pval_adj_thresh) & (df["logfoldchanges"] > logfc_pos_thresh)]
    pos_df = pos_df.sort_values("logfoldchanges", ascending=False).head(200)

    pos_genes_a = pos_df["names"].tolist()

    return (ct_a, ct_b, pos_genes_a, True)


def markers_from_reference(
    adata: AnnData,
    ref_cell_type: str,
    ref_gene_key: str | None = None,
    ref_raw_counts_layer: str | None = "raw",
    mode: str = "de",
    max_fpr: float | None = None,
    auc_pos_thresh: float = 0.9,
    method: str = "wilcoxon",
    pval_adj_thresh: float = 0.05,
    logfc_pos_thresh: float = 1.0,
    vote_fraction_pos: float = 0.5,
    min_pos_frac: float = 0.1,
    max_neg_frac: float = 0.05,
    t_pos: float = 0.25,
    t_neg: float = 1.0,
    min_cells_per_celltype: int = 10,
    n_jobs: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """
    Compute positive and negative markers per cell type using pairwise contrasts
    (AUC/pAUC or DE) followed by voting and a rarity-based definition of
    negative markers.

    Positive markers:
    For each cell type c, a gene g is considered a positive marker if it is
    "up in c" in at least ceil(vote_fraction_pos * M_c) of its valid pairwise
    comparisons (M_c). Additionally, g must be expressed (> 0) in at least
    min_pos_frac fraction of cells of type c in the reference dataset.

    Negative markers:
    For each ordered pair (a, b) of cell types, take genes up in a vs b and consider them
    negative-marker candidates for b if  (1.) they are expressed (> 0) in at most
    max_neg_frac fraction of cells of type b, and (2.) are not up in b vs any cell
    type (computed across all ordered contrasts).

    Overlap filtering:
    Overlap filtering is applied separately to positive and negative markers:
        - Positive lists: genes appearing in ≥ t_pos * n_types lists are dropped.
        - Negative lists: genes appearing in ≥ t_neg * n_types lists are dropped.

    Parameters
    ----------
    adata : AnnData
        Reference single-cell dataset (cells x genes).
    ref_cell_type : str
        Column in `adata.obs` containing cell type labels.
    ref_gene_key : str or None, default=None
        Column in `adata_ref.var` containing gene identifiers.
        If `None`, `adata_ref.var_names` are used.
    ref_raw_counts_layer : str or None, default=None
        Layer containing raw counts. If `None`, raw counts are expected in
        `adata.X`.
    mode : {"auc", "de"}, optional (default: "de")
        - "auc": compute markers using pairwise AUC/pAUC.
        - "de" : compute markers using pairwise DE.
    max_fpr : float or None, optional (default: None)
        (AUC mode only)
        If None, compute full AUC. If in (0, 1], compute standardized pAUC over
        [0, max_fpr] using sklearn's `roc_auc_score(max_fpr=max_fpr)`.
    auc_pos_thresh : float, optional (default: 0.9)
        (AUC mode only)
        Minimum AUC/pAUC for a gene to be considered "up in c_i vs c_j".
    method : str, optional (default: "wilcoxon")
        (DE mode only)
        DE method passed to `sc.tl.rank_genes_groups` ("wilcoxon", "t-test", "logreg", ...).
    pval_adj_thresh : float, optional (default: 0.05)
        (DE mode only)
        FDR (adjusted p-value) cutoff for positive markers.
    logfc_pos_thresh : float, optional (default: 1.0)
        (DE mode only)
        Minimum log fold-change for *positive* markers (c > d).
    vote_fraction_pos : float, optional (default: 0.5)
        Fraction of valid pairwise contrasts in which a gene must be "up in c"
        (AUC mode) / significantly up in c (DE mode) to be called a positive
        marker of c.
    min_pos_frac : float, optional (default: 0.1)
        Minimum fraction of cells of type c in which a gene must be expressed
        (counts > 0) in the reference dataset to be considered a *positive*
        marker of c.
    max_neg_frac : float, optional (default: 0.05)
        Maximum fraction of cells of type c in which a gene may be expressed
        (counts > 0) in the reference dataset to be considered a *negative*
        marker of c.
    t_pos : float, optional (default: 0.25)
        Overlap filter threshold for positive markers.
    t_neg : float, optional (default: 1.0)
        Overlap filter threshold for negative markers.
    min_cells_per_celltype : int, optional (default: 10)
        Minimum number of cells required per cell type to be included in pairwise
        computations.
    n_jobs : int, optional (default: 1)
        Number of parallel jobs for running pairwise computations.

    Returns
    -------
    dict
        A dictionary mapping each cell type to its positive and negative markers:
        {cell_type: {"positive": [genes], "negative": [genes]}}
    """
    # copies gene identifiers into var_names and makes them unique (if needed)
    adata = _make_ref_genes_unique(adata, ref_gene_key=ref_gene_key)

    # getting gene names and mapping to indices for later use
    var_names = adata.var_names
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    # raw counts for expression fraction computation (must be before normalization)
    counts = _get_count_matrix(adata, layer=ref_raw_counts_layer, layer_arg="ref_raw_counts_layer")

    # applying normalization and log1p to get data ready for DE/AUC
    # stored in X directly, since adata was copied previously
    adata = _get_norm_log(adata, layer=ref_raw_counts_layer, layer_arg="ref_raw_counts_layer")
    adata.X = adata.layers[NORM_LOG_LAYER]

    ctypes = pd.Categorical(adata.obs[ref_cell_type])
    types = list(ctypes.categories)
    if len(types) < 2:
        raise ValueError("Need at least two cell types to compute markers.")

    # Cell counts per type -> filter rare cell types from pairwise contrasts
    cell_counts = ctypes.value_counts().to_dict()

    usable_celltypes = [ct for ct in types if cell_counts.get(ct, 0) >= min_cells_per_celltype]
    n_celltypes = len(usable_celltypes)
    if n_celltypes < 2:
        raise ValueError(
            f"Fewer than two cell types have at least {min_cells_per_celltype} cells; "
            f"cannot perform pairwise contrasts."
        )

    # ordered cell type pairs (a, b), a != b
    celltype_pairs = [(ct_a, ct_b) for ct_a in usable_celltypes for ct_b in usable_celltypes if ct_a != ct_b]

    # Precompute fraction of cells with counts > 0 per type
    # This dictionary maps cell type to an array of shape (n_genes,)
    # with the fraction of cells of that type expressing each gene.
    expr_frac: dict[str, np.ndarray] = {}
    for ct in usable_celltypes:
        mask_ct = ctypes == ct
        X_ct = counts[mask_ct]

        if sparse.issparse(X_ct):
            frac = X_ct.getnnz(axis=0) / X_ct.shape[0]
        else:
            frac = (X_ct > 0).mean(axis=0)

        expr_frac[ct] = np.asarray(frac).ravel()

    # Choose worker based on mode
    if mode == "auc":

        def worker(ct_a: str, ct_b: str):
            return _pairwise_auc(
                adata=adata,
                ctypes=ctypes,
                gene_key=ref_gene_key,
                ref_cell_type=ref_cell_type,
                ct_a=ct_a,
                ct_b=ct_b,
                max_fpr=max_fpr,
                auc_pos_thresh=auc_pos_thresh,
                min_cells_per_celltype=min_cells_per_celltype,
            )

    elif mode == "de":

        def worker(ct_a: str, ct_b: str):
            return _pairwise_de(
                adata=adata,
                ctypes=ctypes,
                ref_cell_type=ref_cell_type,
                ct_a=ct_a,
                ct_b=ct_b,
                method=method,
                pval_adj_thresh=pval_adj_thresh,
                logfc_pos_thresh=logfc_pos_thresh,
                min_cells_per_celltype=min_cells_per_celltype,
            )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'auc' or 'de'.")

    # Run over all pairs, possibly in parallel
    if n_jobs == 1:
        results = [worker(ct_a, ct_b) for ct_a, ct_b in celltype_pairs]
    else:
        results = Parallel(n_jobs=n_jobs)(joblib_delayed(worker)(ct_a, ct_b) for ct_a, ct_b in celltype_pairs)

    # the ok column indicates whether the pairwise computation was valid (enough cells, etc.)
    # we filter out invalid pairs before building the up_by_pair dictionary
    pair_df = pd.DataFrame(results, columns=["ct_a", "ct_b", "pos_genes_a", "ok"])
    pair_df = pair_df[pair_df["ok"]].reset_index(drop=True)

    # Dictionary mapping (ct_a, ct_b) -> list of genes up in ct_a vs ct_b
    up_by_pair: dict[tuple[str, str], list[str]] = {
        (row.ct_a, row.ct_b): list(row.pos_genes_a) for row in pair_df.itertuples(index=False)
    }

    # per-cell-type union of all "up" genes
    up_any: dict[str, set[str]] = {ct: set() for ct in usable_celltypes}
    for row in pair_df.itertuples(index=False):
        up_any[row.ct_a].update(row.pos_genes_a)

    # -------------------------------------------------------------------------
    # Aggregate positives by voting; aggregate negatives by union (no vote filter)
    # -------------------------------------------------------------------------
    pos_votes = {ct: collections.Counter() for ct in usable_celltypes}
    pair_counts_pos = {ct: 0 for ct in usable_celltypes}  # valid (ct as "a")
    neg_sets = {ct: set() for ct in usable_celltypes}  # accumulate negatives as a set (union)

    for ct_a, ct_b in celltype_pairs:
        up_ab = up_by_pair.get((ct_a, ct_b), set())
        if not up_ab:
            continue  # no genes passed in this contrast

        # positives: genes up in a vs b and common in a - vote for a
        frac_a = expr_frac[ct_a]
        for g in up_ab:
            idx = gene_to_idx.get(g)
            if idx is None:
                continue
            if frac_a[idx] >= min_pos_frac:
                pos_votes[ct_a][g] += 1
        pair_counts_pos[ct_a] += 1

        # negatives: genes up in a vs b, rare in b, AND not up in b vs any cell type
        frac_b = expr_frac[ct_b]
        up_b_any = up_any.get(ct_b, set())

        for g in up_ab:
            if g in up_b_any:
                continue
            idx = gene_to_idx.get(g)
            if idx is None:
                continue
            if frac_b[idx] <= max_neg_frac and (frac_a[idx] >= min_pos_frac):
                neg_sets[ct_b].add(g)

    # ------------------------------------------------------------
    # Build positive marker lists using per-type voting thresholds
    # ------------------------------------------------------------
    pos_lists: dict[str, list[str]] = {}
    for ct in usable_celltypes:
        M_c = pair_counts_pos.get(ct, 0)
        if M_c == 0:
            pos_lists[ct] = []
            continue

        min_pos_votes = max(1, int(np.ceil(vote_fraction_pos * M_c)))

        pos_genes = [g for g, k in pos_votes[ct].items() if (k >= min_pos_votes)]
        pos_genes = sorted(pos_genes, key=lambda g: pos_votes[ct][g], reverse=True)
        pos_lists[ct] = pos_genes

    # Overlap filter for positive markers
    pos_lists = _apply_overlap_filter(pos_lists, t=t_pos, n_ct=n_celltypes)

    # ------------------------------------------------------------
    # Build negative marker lists
    # ------------------------------------------------------------
    # Keep negative markers only if they are positive markers of at least one other cell type
    pos_any_final: set[str] = set().union(*pos_lists.values()) if len(pos_lists) else set()
    for ct in usable_celltypes:
        neg_sets[ct] = {g for g in neg_sets[ct] if g in pos_any_final}

    neg_lists: dict[str, list[str]] = {ct: sorted(list(neg_sets[ct])) for ct in usable_celltypes}

    # Overlap filter for negative markers
    neg_lists = _apply_overlap_filter(neg_lists, t=t_neg, n_ct=n_celltypes)

    markers: dict[str, dict[str, list[str]]] = {
        ct: {"positive": pos_lists.get(ct, []), "negative": neg_lists.get(ct, [])} for ct in usable_celltypes
    }

    return markers


def _is_missing(x):
    """Return True for any kind of NA / NaN / None."""
    try:
        # Works for np.nan, float('nan'), pd.NA, pd.NaT, None
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def _ensure_index(
    gdf,
    *,
    shapes_key: str,
    id_key_name: str,
    id_key: str,
):
    """
    Ensure `gdf` is indexed by `id_key`.

    If `id_key` matches the current index name, the GeoDataFrame is returned unchanged.
    Otherwise, `id_key` must be a column name and will be set as the index.

    If the chosen IDs contain duplicates, the index is reset to a unique RangeIndex.
    """
    if id_key is None:
        # if the ID key is None, we set the index name to "segtraq_id" (if not already set) and use the index as IDs
        if gdf.index.name not in [None, "segtraq_id"]:
            raise ValueError(
                f"You set {id_key_name} to None, but the index of shapes '{shapes_key}' has a name '{gdf.index.name}'. "
                f"Please set {id_key_name}='{gdf.index.name}' instead."
            )
        else:
            warnings.warn(
                f"The dataframe for shapes '{shapes_key}' has no index name. "
                f"Setting index name to {SEGTRAQ_CELL_ID_KEY}.",
                UserWarning,
                stacklevel=2,
            )
            gdf.index.name = SEGTRAQ_CELL_ID_KEY
            id_key = gdf.index.name

    if gdf.index.name == id_key:
        if gdf.index.has_duplicates:
            warnings.warn(
                f"Duplicate IDs detected in index '{id_key}' for shapes '{shapes_key}'. "
                f"Resetting and renaming index to `{SEGTRAQ_CELL_ID_KEY}` to ensure uniqueness.",
                UserWarning,
                stacklevel=2,
            )
            if gdf.index.name in gdf.columns:
                gdf = gdf.drop(columns=[gdf.index.name])
            gdf = gdf.reset_index(drop=False)
            gdf.index.name = SEGTRAQ_CELL_ID_KEY
        return gdf

    if id_key not in gdf.columns:
        raise KeyError(
            f"'{id_key}' not found in shapes '{shapes_key}'. "
            f"Available columns: {gdf.columns.tolist()}. "
            f"Provide a valid {id_key_name}, or set it to the current "
            f"index name ({gdf.index.name}) if IDs are in the index."
        )

    if gdf[id_key].duplicated().any():
        warnings.warn(
            f"Duplicate IDs detected in column '{id_key}' for shapes '{shapes_key}'. "
            f"Instead of using {id_key} as index, resetting the current index and "
            f"renaming it to `{SEGTRAQ_CELL_ID_KEY}`.",
            UserWarning,
            stacklevel=2,
        )
        if gdf.index.name in gdf.columns:
            gdf = gdf.drop(columns=[gdf.index.name])
        gdf = gdf.reset_index(drop=False)
        gdf.index.name = SEGTRAQ_CELL_ID_KEY
        return gdf

    warnings.warn(
        f"Setting column '{id_key}' as the index for shapes '{shapes_key}', "
        "as this is required to link the table to shapes in SpatialData.",
        UserWarning,
        stacklevel=2,
    )
    return gdf.set_index(id_key, drop=True)


def bins_to_transcripts(
    sdata: sd.SpatialData,
    tables_key: str,
    cell_shapes_key: str,
    tables_gene_key: str | None = None,
    bins_shapes_key: str | None = None,
    coordinate_system: str | None = None,
    bins_points_key: str | None = None,
    cell_id_key: str = "cell_id",
    background_id: str | int = "UNASSIGNED",
    chunk_bins: int = 50_000,
) -> sd.SpatialData:
    """
    Convert per-bin/spot counts in sdata.tables[table_key] into per-transcript points.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing tables, shapes and/or points layers.
    tables_key : str
        Key in `sdata.tables` containing the per-bin or per-spot count matrix
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    cell_shapes_key : str
        Key in `sdata.shapes` containing cell segmentation polygons used to
        assign each bin/spot to a `cell_id`.
    bins_shapes_key : str or None, optional
        Key in `sdata.shapes` describing bin/spot geometries. If provided,
        centroids will be computed from these shapes. Exactly one of
        `bins_shapes_key` or `bins_points_key` must be given.
    coordinate_system : str or None, optional
        Coordinate system used when computing centroids from `bins_shapes_key`.
        Required if `bins_shapes_key` is provided.
    bins_points_key : str or None, optional
        Key in `sdata.points` containing precomputed bin/spot centroids with
        x/y coordinates. Used instead of computing centroids from shapes.
    cell_id_key : str, default="cell_id"
        Column or index name in `sdata.shapes[cell_shapes_key]` identifying
        individual cells.
    background_id : str or int, default="UNASSIGNED"
        Identifier assigned to bins/spots that do not intersect any cell.
    chunk_bins : int, default=50_000
        Number of bins processed per chunk when expanding counts into
        transcripts. Smaller values reduce peak memory usage but may increase
        runtime.

    Returns
    -------
    SpatialData
        Updated `SpatialData` object where per-bin counts have been expanded
        into a transcript-level points layer.

    Requirements
    ------------
    - sdata.tables[table_key] is AnnData-like with X = counts (n_bins x n_genes).
    - You can provide either:
        (A) bins_shapes_key (+ coordinate_system) to compute centroids, OR
        (B) bins_points_key containing x/y for each bin/spot.
    - cell_shapes_key contains cell polygons with a column (or index) cell_id_key.
    """

    if (bins_shapes_key is None) == (bins_points_key is None):
        raise ValueError("Provide exactly one of bins_shapes_key or bins_points_key.")

    adata = sdata.tables[tables_key]
    if not sparse.issparse(adata.X):
        X = sparse.csr_matrix(adata.X)
    else:
        X = adata.X.tocsr()

    genes = _get_genes(adata, tables_gene_key)
    gene_names = np.asarray(genes)

    # build centroid points for bins/spots
    if bins_points_key is None:
        if coordinate_system is None:
            raise ValueError("coordinate_system is required when computing centroids from shapes.")

        centroids = sd.get_centroids(
            sdata.shapes[bins_shapes_key],
            coordinate_system=coordinate_system,
        )
        # copy transforms so points align with images/shapes
        centroids.attrs["transform"] = sdata.shapes[bins_shapes_key].attrs.get("transform", None)

        # save in sdata.points under a new key
        bins_points_key = f"{bins_shapes_key}_centroids"
        sdata.points[bins_points_key] = centroids

    cent = sdata.points[bins_points_key]

    # Dask dataframe - compute only necessary cols once
    cent_pd = cent[["x", "y"]].compute()

    # ensure same order
    if "location_id" in adata.obs.columns:
        cent_pd = cent_pd.reindex(adata.obs["location_id"].to_numpy())
    else:
        cent_pd = cent_pd.reindex(adata.obs_names)

    if cent_pd[["x", "y"]].isna().any().any():
        raise ValueError(
            "Centroid x/y contains NaNs after alignment. Check bins_points/bins_shapes vs table row identifiers."
        )

    x_all = cent_pd["x"].to_numpy(dtype=np.float32, copy=False)
    y_all = cent_pd["y"].to_numpy(dtype=np.float32, copy=False)

    # assign each bin/spot to a cell_id

    points_gdf = gpd.GeoDataFrame(  # spatial join
        cent_pd.copy(),
        geometry=gpd.points_from_xy(cent_pd["x"], cent_pd["y"]),
    )

    cells = sdata.shapes[cell_shapes_key]

    # ensure cell_id_key exists as column (if it's in index, expose it)
    cells_gdf = cells[["geometry"]].copy()
    if cell_id_key in cells.columns:
        cells_gdf[cell_id_key] = cells[cell_id_key].values
    elif cells.index.name == cell_id_key:
        cells_gdf[cell_id_key] = cells.index.values
    else:
        raise ValueError(
            f"cell_id_key={cell_id_key!r} not found as a column or index name in sdata.shapes[{cell_shapes_key!r}]."
        )

    joined = gpd.sjoin(
        points_gdf[["geometry"]],
        cells_gdf,
        how="left",
        predicate="intersects",
    )

    cell_id_series = (
        joined[cell_id_key].groupby(level=0).first().reindex(points_gdf.index).fillna(background_id)  # centroid index
    )

    # expand sparse counts -> per-transcript rows (x, y, gene, cell_id)
    cell_cat = cell_id_series.astype("category")  # categorical codes (memory-friendly)
    cell_codes = cell_cat.cat.codes.to_numpy(dtype=np.int32, copy=False)
    cell_categories = cell_cat.cat.categories.to_numpy()

    @delayed
    def chunk_to_molecules(start: int, end: int) -> pd.DataFrame:
        Xc = X[start:end].tocoo()
        if Xc.nnz == 0:
            return pd.DataFrame({"x": [], "y": [], "feature_name": [], "cell_id": []})

        bin_idx = (Xc.row + start).astype(np.int64, copy=False)
        gene_idx = Xc.col.astype(np.int32, copy=False)
        counts = Xc.data.astype(np.int32, copy=False)

        # expand each (bin,gene,count) into `count` molecules
        bin_rep = np.repeat(bin_idx, counts)
        gene_rep = np.repeat(gene_idx, counts)

        return pd.DataFrame(
            {
                "x": x_all[bin_rep],
                "y": y_all[bin_rep],
                "feature_name": gene_names[gene_rep].astype(object),
                "cell_id": cell_categories[cell_codes[bin_rep]].astype(object),
            }
        )

    meta = pd.DataFrame(
        {
            "x": pd.Series(dtype="float32"),
            "y": pd.Series(dtype="float32"),
            "feature_name": pd.Series(dtype="object"),
            "cell_id": pd.Series(dtype="object"),
        }
    )

    n_bins = X.shape[0]
    parts = [chunk_to_molecules(start, min(start + chunk_bins, n_bins)) for start in range(0, n_bins, chunk_bins)]
    molecules_ddf = dd.from_delayed(parts, meta=meta)
    # molecules_ddf = molecules_ddf.reset_index(drop=True)

    # wrap as SpatialData points with transforms
    transforms = cent.attrs.get("transform", None)

    molecules_points = PointsModel.parse(
        molecules_ddf,
        feature_key="feature_name",
        transformations=transforms,
    )

    sdata.points["transcripts"] = molecules_points

    return sdata


def validate_spatialdata(
    sdata: sd.SpatialData,
    images_key: str | None = "morphology_focus",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_key: str | None = "cell_area",
    tables_centroid_x_key: str | None = "x_centroid",
    tables_centroid_y_key: str | None = "y_centroid",
    tables_gene_key: str | None = None,
    tables_raw_counts_layer: str | None = None,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: str = "UNASSIGNED",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_z_key: str | None = "z",
    points_gene_key: str = "feature_name",
    shapes_key: str | list[str] = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    nucleus_shapes_key: str | None = "nucleus_boundaries",
    nucleus_shapes_cell_id_key: str = "cell_id",
) -> bool:
    """
    Validates the integrity of a SpatialData object by checking the consistency of cell IDs
    across points, shapes, and tables.

    This function ensures that:
    - All points have corresponding shapes, and tables.
    - Cell IDs in points match those in shapes, and tables.
    - If shapes are present, they contain all cell IDs from the points.
    - If tables are present, they contain all cell IDs from the shapes.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object to validate.
    tables_key : str, optional
        Key for accessing tables in the SpatialData. Default is "table".
    tables_cell_id_key : str, optional
        Column name in the tables DataFrame (AnnData.obs) that contains cell IDs. Default is "cell_id".
    tables_area_key : str or None, optional, default="cell_area"
        Column in the cell table with cell area (2D).
        If `None`, area/volume-based metrics will be computed via `segtraq.bl.morphological_features`.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    tables_raw_counts_layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts.
        If a layer is specified, it must exist and contain count-like values.
    points_key : str, optional
        Key for accessing points (e.g., transcripts) in the SpatialData. Default is "transcripts".
    points_cell_id_key : str, optional
        Column name in the points DataFrame indicating cell assignments. Default is "cell_id".
    points_background_id : str, optional
        Identifier used for unassigned or background transcripts in the points DataFrame. Default is "UNASSIGNED".
    shapes_key : str or list of str, optional
        Key(s) for accessing shapes (e.g., cell boundaries) in the SpatialData. Default is "cell_boundaries".
        Can be a list if multiple shape layers are present.
    shapes_cell_id_key : str, optional, default="cell_id"
        Cell ID key for `sdata.shapes[shapes_key]`. Must match either the shapes index name
        or a column name (which will be set as the index if needed).
        If `None`, the index is assumed to contain cell IDs and
        renamed to "segtraq_cell_id".
    nucleus_shapes_key : str or None, optional, default="nucleus_boundaries"
        Key in `sdata.shapes` for nucleus boundary polygons, if available.
        If None, a nucleus mask can be obtained via `segtraq.run_cellpose`.
    nucleus_shapes_cell_id_key : str, optional, default="cell_id"
        Cell ID key for `sdata.shapes[nucleus_shapes_key]`. Must match either the shapes
        index name or a column name (which will be set as the index if needed).

    Raises
    ------
    TypeError
        If the input is not an instance of sd.SpatialData.
    ValueError
        If the SpatialData object does not contain points or if there are inconsistencies in cell IDs.

    Returns
    -------
    bool
        True if the SpatialData object passes all validation checks. Otherwise, an error or warning is raised.
    """
    if not isinstance(sdata, sd.SpatialData):
        raise TypeError("Input must be an instance of sd.SpatialData")

    # check if there is an image at the specified key
    if images_key is not None:
        assert images_key in sdata.images.keys(), (
            f"{images_key} not found in the image layer. "
            f"Available keys: {sdata.images.keys()}. "
            "You can set this with the images_key parameter (set to None if you do not have this)."
        )

    contains_points = len(sdata.points) > 0
    contains_shapes = len(sdata.shapes) > 0
    contains_tables = len(sdata.tables) > 0

    # check if there are points in the spatial data
    if not contains_points:
        raise ValueError("SpatialData object must contain points (transcripts)")

    # get the cell IDs from the points
    assert points_key in sdata.points, (
        f"SpatialData must contain points with key: {points_key}. "
        f"Available keys: {list(sdata.points.keys())}. "
        f"You can set this with the 'points_key' argument."
    )
    points = sdata.points[points_key]

    # check gene column in points
    assert points_cell_id_key in points.columns, (
        f"Points DataFrame must contain column to identify cells: {points_cell_id_key}. "
        f"Available columns: {points.columns.tolist()}. "
        f"You can set this with the 'points_cell_id_key' argument."
    )

    # check coordinate columns in points
    for coord_key, arg_name in [
        (points_x_key, "points_x_key"),
        (points_y_key, "points_y_key"),
    ]:
        assert coord_key in points.columns, (
            f"Points DataFrame must contain coordinate column '{coord_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the '{arg_name}' argument."
        )

    if points_z_key is not None:
        assert points_z_key in points.columns, (
            f"Points DataFrame must contain z coordinate column '{points_z_key}'. "
            f"Available columns: {points.columns.tolist()}. "
            f"You can set this with the 'points_z_key' argument. Set to None if you do not have z coordinates."
        )

    # check gene key
    assert points_gene_key in points.columns, (
        f"Points DataFrame must contain gene feature column '{points_gene_key}'. "
        f"Available columns: {points.columns.tolist()}. "
        f"You can set this with the 'points_gene_key' argument."
    )

    # get unique cell IDs from points
    points_df = points.compute() if hasattr(points, "compute") else points  # precompute is faster
    transcript_ids = set(points_df[points_cell_id_key].unique())
    shapes_cell_ids = set()

    # if there are shapes, ensure that there are no cell IDs in the points that are not in the shapes
    if contains_shapes:
        # we can have multiple shape keys (e. g. when using multiple layers in proseg), so we need to handle them here
        if isinstance(shapes_key, str):
            assert shapes_key in sdata.shapes, (
                f"Shapes DataFrame must contain key: {shapes_key}. "
                f"Available keys: {list(sdata.shapes.keys())}. "
                f"If you want to use a different key, you can set this with the 'shapes_key' argument."
            )
            shapes = sdata.shapes[shapes_key]

            # this ensures that the index of the shapes df is always the cell ID
            shapes = _ensure_index(
                shapes, shapes_key=shapes_key, id_key=shapes_cell_id_key, id_key_name="shapes_cell_id_key"
            )
            # setting the coordinate system to None to avoid issues when computing cell area
            # or other morphological features later on
            shapes = shapes.set_crs(None, allow_override=True)
            sdata.shapes[shapes_key] = shapes
            shapes_cell_ids = set(shapes.index.tolist())
        else:
            raise ValueError("shapes_key must be a string or a list of strings")

        # ensuring that all cell IDs have the same dtype (either str or numeric)
        # taking a random ID from each set and comparing dtypes
        transcript_sample = next(iter(transcript_ids))
        shapes_sample = next(iter(shapes_cell_ids))

        def is_numeric(x):
            return isinstance(x, int | float | np.integer | np.floating)

        def is_string(x):
            return isinstance(x, str)

        if (is_numeric(transcript_sample) and is_numeric(shapes_sample)) or (
            is_string(transcript_sample) and is_string(shapes_sample)
        ):
            pass  # OK, both numeric or both string
        else:
            raise TypeError(
                f"Cell ID types between points and shapes are incompatible: "
                f"{type(transcript_sample)} (points) vs {type(shapes_sample)} (shapes). "
                f"Please ensure that cell IDs are all strings or all numeric."
            )

        # if the user provided a background ID, we want to ensure that it actually occurs
        if points_background_id is not None:
            assert points_background_id in transcript_ids, (
                f"points_background_id '{points_background_id}' not found among point cell IDs. "
                f"You can set this with the 'points_background_id' argument. "
                f"If you do not have a background ID, set this parameter to None."
            )

            # as a more stringent check, we also raise a warning if the background ID is not the most common one
            most_common_points_id = points_df[points_cell_id_key].mode().iloc[0]
            if most_common_points_id != points_background_id:
                warnings.warn(
                    f"points_background_id '{points_background_id}' is not the most common cell ID "
                    f"among points (most common is '{most_common_points_id}'). "
                    "This may indicate that your background ID is not correctly set. "
                    "If you are sure that 'points_background_id' is correct, you can ignore this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        # missing_in_polygons = { #TODO - after querying sdata objects, this breaks
        #     x
        #     for x in (transcript_ids - shapes_cell_ids - {points_background_id})
        #     if not _is_missing(x)  # also removing any NAs (no matter if from pandas, np, or None)
        # }
        # assert len(missing_in_polygons) == 0, (
        #     f"Missing {len(missing_in_polygons)} cell IDs from polygons: "
        #     f"{list(missing_in_polygons)[: min(5, len(missing_in_polygons))]}... "
        #     f"These cell IDs are present in the points, but not in the shapes. "
        #     f"If your missing cell ID is indicating an unassigned transcript, "
        #     f"you can set the points_background_id parameter."
        # )

        # if shapes and tables are present, ensure that the cell IDs match
        # checking that the adata and the polygons have the same cell IDs
        if contains_tables:
            assert tables_key in sdata.tables, (
                f"Tables DataFrame must contain key: {tables_key}. "
                f"Available keys: {list(sdata.tables.keys())}. "
                f"If you want to use a different key, set the 'tables_key' parameter."
            )
            table = sdata.tables[tables_key]
            # checking if the tables_cell_id_key is a column or an index name,
            # and turning it into a column if it's an index
            _resolve_obs_index_ambiguity(sdata, tables_key, tables_cell_id_key)
            assert tables_cell_id_key in table.obs.columns, (
                f"Tables DataFrame must contain column: {tables_cell_id_key}. "
                f"Available columns: {table.obs.columns.tolist()}. "
                f"If you want to use a different column, set the 'tables_cell_id_key' parameter."
            )

            _check_if_raw = _get_count_matrix(
                sdata.tables[tables_key], tables_raw_counts_layer, layer_arg="tables_raw_counts_layer"
            )

            assert "spatialdata_attrs" in table.uns, "Could not find 'spatialdata_attrs' in table.uns. "
            "You can set them like this: \n"
            "sdata.tables['table'].obs['region'] = 'cell_boundaries'\n"
            "sdata.set_table_annotates_spatialelement('table', region='cell_boundaries')"

            tables_cell_ids = set(table.obs[tables_cell_id_key].values)

            # checking that the dtype of the table cell IDs matches that of the shapes
            table_dtype = table.obs[tables_cell_id_key].dtype
            shapes_sample = next(iter(shapes_cell_ids))
            if is_numeric(shapes_sample):
                if not np.issubdtype(table_dtype, np.number):
                    raise TypeError(
                        f"Cell ID types between shapes and tables are incompatible: "
                        f"{type(shapes_sample)} (shapes) vs {table_dtype} (tables). "
                        f"Please ensure that cell IDs are all strings or all numeric."
                    )
            elif is_string(shapes_sample):
                if not np.issubdtype(table_dtype, np.object_) and not np.issubdtype(table_dtype, np.str_):
                    raise TypeError(
                        f"Cell ID types between shapes and tables are incompatible: "
                        f"{type(shapes_sample)} (shapes) vs {table_dtype} (tables). "
                        f"Please ensure that cell IDs are all strings or all numeric."
                    )

            # --- Ensure consistent types between shapes and tables ---
            # Ignore missing values (e.g. NaN, None) when checking type
            non_missing_shapes = [x for x in shapes_cell_ids if not _is_missing(x)]
            non_missing_tables = [x for x in tables_cell_ids if not _is_missing(x)]

            # Determine dominant type (str or numeric)
            shapes_has_str = any(isinstance(x, str) for x in non_missing_shapes)
            tables_has_str = any(isinstance(x, str) for x in non_missing_tables)

            # If one side contains strings, convert both sides to string
            if shapes_has_str or tables_has_str:
                shapes_cell_ids = {str(x) for x in shapes_cell_ids if not _is_missing(x)}
                tables_cell_ids = {str(x) for x in tables_cell_ids if not _is_missing(x)}
                points_background_id = str(points_background_id)
            else:
                # Ensure we drop any NAs (NaN, None, etc.) before comparison
                shapes_cell_ids = {x for x in shapes_cell_ids if not _is_missing(x)}
                tables_cell_ids = {x for x in tables_cell_ids if not _is_missing(x)}

            # --- Perform set comparisons ---
            missing_in_shapes = tables_cell_ids - shapes_cell_ids - {points_background_id}
            missing_in_tables = shapes_cell_ids - tables_cell_ids - {points_background_id}

            if len(missing_in_tables) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_tables)} cell IDs in tables: "
                    f"{list(missing_in_tables)[: min(5, len(missing_in_tables))]}... "
                    "These cells are present in shapes, but not in tables. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
                )
            if len(missing_in_shapes) != 0:
                warnings.warn(
                    f"Missing {len(missing_in_shapes)} cell IDs in shapes: "
                    f"{list(missing_in_shapes)[: min(5, len(missing_in_shapes))]}... "
                    "These cells are present in tables, but not in shapes. "
                    "This might lead to inconsistencies in the spatialdata object.",
                    stacklevel=2,
                )

            # check that gene names in the table are compatible with those in the points
            genes_in_points = set(points_df[points_gene_key].unique())  # faster
            genes_in_table = set(_get_genes(table, tables_gene_key))
            common_genes = genes_in_points & genes_in_table
            if len(common_genes) == 0:
                raise ValueError(
                    "No common genes found between points and tables. "
                    "Please ensure that the gene names in both are compatible. "
                    f"Genes in points: {list(genes_in_points)[:5]}..., "
                    f"Genes in tables: {list(genes_in_table)[:5]}..."
                )

            if tables_area_key is not None:
                assert tables_area_key in table.obs.columns, (
                    f"Tables DataFrame must contain area/volume column '{tables_area_key}'. "
                    f"Available columns: {table.obs.columns.tolist()}. "
                    f"You can set this with the 'tables_area_key' argument (set to None if you do not have this)."
                )
            if tables_area_key is None:
                warnings.warn(
                    "No area column specified for tables. Area will be automatically computed from shapes.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                bl.morphological_features(
                    sdata,
                    features_to_compute=["cell_area"],
                    tables_cell_id_key=tables_cell_id_key,
                    tables_centroid_x_key=tables_centroid_x_key,
                    tables_centroid_y_key=tables_centroid_y_key,
                    shapes_key=shapes_key,
                    tables_key=tables_key,
                    inplace=True,
                )

            if tables_centroid_x_key is not None:
                assert tables_centroid_x_key in table.obs.columns, (
                    f"Tables DataFrame must contain x coordinate column '{tables_centroid_x_key}'. "
                    f"Available columns: {table.obs.columns.tolist()}. "
                    f"You can set this with the 'tables_centroid_x_key' argument (set to None if you do not have this)."
                )

            if tables_centroid_y_key is not None:
                assert tables_centroid_y_key in table.obs.columns, (
                    f"Tables DataFrame must contain y coordinate column '{tables_centroid_y_key}'. "
                    f"Available columns: {table.obs.columns.tolist()}. "
                    f"You can set this with the 'tables_centroid_y_key' argument (set to None if you do not have this)."
                )

            if tables_centroid_x_key is None or tables_centroid_y_key is None:
                warnings.warn(
                    "No centroids specified for tables. Centroids will be automatically computed from shapes.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                bl.morphological_features(
                    sdata,
                    tables_cell_id_key=tables_cell_id_key,
                    tables_centroid_x_key=tables_centroid_x_key,
                    tables_centroid_y_key=tables_centroid_y_key,
                    shapes_key=shapes_key,
                    features_to_compute=["centroid"],
                    tables_key=tables_key,
                    inplace=True,
                )
        else:
            raise ValueError("SpatialData object must contain a table.")
    else:
        raise ValueError("SpatialData object must contain shapes.")

    # Check nucleus shapes
    if nucleus_shapes_key is not None:
        if nucleus_shapes_key not in sdata.shapes:
            raise KeyError(
                f"Nucleus shapes key '{nucleus_shapes_key}' not found in sdata.shapes. "
                f"Available keys: {list(sdata.shapes.keys())}."
                f"You can set this with the 'nucleus_shapes_key' argument. "
                f"Set to None if you do not have nucleus shapes."
            )

        nucleus_shapes = sdata.shapes[nucleus_shapes_key]
        nucleus_shapes = _ensure_index(
            nucleus_shapes,
            shapes_key=nucleus_shapes_key,
            id_key=nucleus_shapes_cell_id_key,
            id_key_name="nucleus_shapes_cell_id_key",
        )
        # setting the coordinate system to None to avoid issues when computing cell area
        # or other morphological features later on
        nucleus_shapes = nucleus_shapes.set_crs(None, allow_override=True)
        sdata.shapes[nucleus_shapes_key] = nucleus_shapes

    return True


def _process_image(
    sdata: sd.SpatialData,
    channel: str = "DAPI",
    images_key: str = "morphology_focus",
    images_data_key: str = "scale0/image",
    key_added: str = "nucleus_boundaries",
    return_values: bool = True,
):
    if key_added is not None:
        assert key_added not in sdata.labels.keys(), (
            f"Key {key_added} already exists in spatial data object. Please choose another key."
        )

    image = sdata.images[images_key]

    if isinstance(image, xr.DataTree):
        assert images_data_key is not None, (
            f"It looks like your image is stored as a DataTree. "
            f"Please provide a data_key to access the image data. "
            f"Available keys are: {list(image.keys())}."
        )
        assert images_data_key.split("/")[0] in image.keys(), (
            f"Data key {images_data_key} not found in the image data. Available keys: {list(image.keys())}"
        )

        image = image[images_data_key]

        assert isinstance(image, xr.DataArray), (
            f"The image data should be a DataArray. "
            f"Please provide a valid data key. "
            f"Available keys are: {[images_data_key + '/' + x for x in list(image.keys())]}."
        )

    try:
        image = image.sel(c=[channel])
    except KeyError as err:
        raise KeyError(
            f"Channel {[channel]} not found in the image data. Available channels: {list(image.c.values)}"
        ) from err

    if return_values:
        # returning a numpy array
        return image.values
    # returning an xarray object
    return image


def _cellpose(
    img: np.ndarray,
    diameter: float = None,
    channel_settings: list = None,
    num_iterations: int = 2000,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    batch_size: int = 8,
    gpu: bool = True,
    model_type: str = "cyto3",  # cellpose < 4.0
    pretrained_model: str = "cpsam",  # cellpose 4.0
    postprocess_func: Callable = lambda x: x,
    **kwargs,
):
    from cellpose import models

    cp_version = version("cellpose")

    # checking that the input is 2D or 3D
    if img.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, got {img.ndim}.")

    # if the input is 2D, we add a channel dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    # The cellpose API has changed in version 4.0, so we need to check the version

    if pkg_version.parse(cp_version).major < 4 and channel_settings != [0, 0]:
        assert channel_settings is not None, (
            "The argument channel_settings must be provided for Cellpose < 4.0. "
            "For independent segmentation of each channel, set it to [0, 0]. "
            "For joint segmentation, set it to [1, 2] or [2, 1]."
        )
        assert img.shape[0] == 2, (
            f"Joint segmentation requires exactly two channels. "
            f"You set channel_settings to {channel_settings}, "
            f"but provided {img.shape[0]} channels in the object."
        )
        model = models.Cellpose(gpu=gpu, model_type=model_type)
    else:
        # model_type is not used in cellpose 4.0
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    all_masks = []
    # if the channels are [0, 0], independent segmentation is performed on all channels
    if channel_settings == [0, 0]:
        if img.shape[0] > 1:
            warnings.warn(
                "Performing independent segmentation on all markers. "
                "If you want to perform joint segmentation, "
                "please set the channel_settings argument appropriately.",
                RuntimeWarning,
                stacklevel=2,
            )
        for ch in range(img.shape[0]):
            # Build version-aware keyword arguments
            eval_kwargs = dict(
                diameter=diameter,
                niter=num_iterations,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                batch_size=batch_size,
                **kwargs,
            )

            if pkg_version.parse(cp_version).major < 4:
                eval_kwargs["channels"] = channel_settings

            # Get the image at the channel and run Cellpose
            output = model.eval(img[ch].squeeze(), **eval_kwargs)

            # Unpack outputs based on version
            if pkg_version.parse(cp_version).major >= 4:
                masks_pred, _, diams = output
            else:
                masks_pred, _, _, diams = output

            masks_pred = postprocess_func(masks_pred)
            all_masks.append(masks_pred)
    else:
        # if the channels are anything else, joint segmentation is attempted
        eval_kwargs = dict(
            diameter=diameter,
            niter=num_iterations,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            **kwargs,
        )

        if pkg_version.parse(cp_version).major < 4:
            eval_kwargs["channels"] = channel_settings

        output = model.eval(img.squeeze(), **eval_kwargs)

        # Unpack based on version
        if pkg_version.parse(cp_version).major >= 4:
            masks_pred, _, diams = output
        else:
            masks_pred, _, _, diams = output

        masks_pred = postprocess_func(masks_pred)
        all_masks.append(masks_pred)

    return np.array(all_masks), diams


def _labels_to_shapes(label_img: np.ndarray, simplify_tolerance: float | None = 0.5) -> gpd.GeoDataFrame:
    """
    Convert a 2D label image into polygon boundaries.

    Each connected label is represented by one Polygon or MultiPolygon.

    Parameters
    ----------
    label_img : np.ndarray
        2D array of integer labels. Background should be 0.
    simplify_tolerance : float or None, default=0.5
        Simplification tolerance for polygon boundaries. Set to None or 0
        to disable simplification.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns ["cell_id", "geometry"]. Each row corresponds to one labeled region.
    """
    if label_img.ndim != 2:
        raise ValueError("Input label_img must be 2D.")

    mask = (label_img != 0).astype(np.uint8)
    geometries = []
    cell_ids = []

    # shapes() only accepts either of these dtypes: int16, int32, uint8, uint16, float32, float64, int8
    # if our labels are in uint32, and we have label_img.max() < 2_147_483_647, we need to convert to int32
    if label_img.dtype == np.uint32 and label_img.max() < np.iinfo(np.int32).max:
        label_img = label_img.astype(np.int32)

    for geom, value in shapes(label_img, mask=mask, connectivity=8):
        if value == 0:
            continue
        poly = shape(geom)
        # this turns disconnected polygons into MultiPolygons (which are valid)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.area == 0:
            continue
        if simplify_tolerance and simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        geometries.append(poly)
        cell_ids.append(int(value))

    gdf = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": geometries})
    # Merge multiple geometries per label into a single MultiPolygon
    gdf = gdf.dissolve(by="cell_id", as_index=True)
    # rasterio operates on the pixel centroids instead of the corners
    # to get sub-pixel accurate geometries, we need to shift the geometries by -0.5 in x and y
    gdf["geometry"] = gdf["geometry"].apply(lambda p: translate(p, xoff=-0.5, yoff=-0.5))

    return gdf


def cellpose(
    sdata: sd.SpatialData,
    channel: str = "DAPI",
    images_key: str = "morphology_focus",
    images_data_key: str = "scale0/image",
    shapes_key: str = "cell_boundaries",
    key_added: str = "nucleus_boundaries",
    inplace: bool = True,
    **kwargs,
):
    """
    This function runs the cellpose segmentation algorithm on the provided image data.
    It extracts the image data from the spatialdata object, applies the cellpose algorithm,
    and adds the segmentation masks to the spatialdata object.
    The segmentation masks are stored as polygons in the shapes attribute of the spatialdata object.

    Parameters
    ----------
    sdata : SpatialData
        A `SpatialData` object containing segmented and transcript-assigned spatial
        transcriptomics data (images, tables, points, shapes and optional labels).

    channel: str, default="DAPI"
        The channel(s) to be used for segmentation.

    images_key : str, default="morphology_focus"
        Key in `sdata.images` for a nuclear or morphology image (e.g., DAPI).
        Used for segmentation.

    images_data_key : str, default="scale0/image"
        Key for accessing data in `sdata.images` if they are stored as a DataTree.
        Consider using a higher scale (lower resolution) for segmentation to
        speed up computation and reduce memory usage during Cellpose.

    shapes_key : str, default="cell_boundaries"
        Key in `sdata.shapes` for cell boundary polygons. Used to get transformations.

    key_added: str, default="nucleus_boundaries"
        The key under which the segmentation masks will be stored in the shapes attribute
        of the spatialdata object. Defaults to "nucleus_boundaries".

    inplace, bool, default=True
        Whether to modify the spatialdata object in place. Defaults to True.

    **kwargs: Additional keyword arguments to be passed to the cellpose algorithm.
    """

    if not inplace:
        sdata = sd.deepcopy(sdata)

    # assert that the format is correct and extract the image
    image = _process_image(
        sdata, channel=channel, images_key=images_key, key_added=key_added, images_data_key=images_data_key
    )

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # convert labels to shapes
    nuc_shapes = _labels_to_shapes(segmentation_masks[0])

    # get transformations
    S = get_transformation(sdata.images[images_key][images_data_key]).to_affine_matrix(
        ("x", "y"), ("x", "y")
    )  # get scaling factors
    T = get_transformation_between_coordinate_systems(
        sdata, sdata.images[images_key], sdata.shapes[shapes_key]
    ).to_affine_matrix(("x", "y"), ("x", "y"))  # get affine transformation between image and shapes
    A = T @ S
    t_params = [A[0, 0], A[0, 1], A[1, 0], A[1, 1], A[0, 2], A[1, 2]]

    # apply affine transformation to nuclear shapes to have them in the same coordinate system as cell shapes
    nuc_shapes["geometry"] = nuc_shapes["geometry"].apply(lambda g: affine_transform(g, t_params))

    sdata.shapes[key_added] = sd.models.ShapesModel.parse(nuc_shapes, transformations=None)

    # set transformation for nucleus shapes to be the same as for cell shapes
    cell_shape_transformation = get_transformation(sdata.shapes[shapes_key])
    set_transformation(sdata.shapes[key_added], cell_shape_transformation)

    if not inplace:
        return sdata


def _is_background(series: pd.Series, background_id):
    """
    Checks if values in a Pandas Series match the background_id,
    handling None, np.nan, or pd.NA correctly.

    Args:
        series (pd.Series): The column data (e.g., 'cell_id').
        background_id: The value representing background (e.g., 'UNASSIGNED', 0, np.nan).

    Returns:
        pd.Series (bool): A boolean Series (True if background, False otherwise).
    """
    # Use pd.isna() to reliably check for None, np.nan, or pd.NA in background_id
    if pd.isna(background_id):
        # If the background ID is a null value, check for missing data in the Series
        is_background = series.isna()
    else:
        # Otherwise, perform a direct equality check
        is_background = series == background_id

    return is_background


def filter_cells(adata, col: str, func: Callable):
    """
    Filter cells in an AnnData object based on a condition applied to a specified column.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing cell data.
    col : str
        The column name in the cell metadata to apply the filter on.
    func : Callable
        A function that takes a Pandas Series (column data) and returns a boolean mask.

    Returns
    -------
    AnnData
        A new AnnData object containing only the cells that satisfy the condition.
    """
    assert col in adata.obs.columns, (
        f"Column '{col}' not found in adata.obs. Available columns: {adata.obs.columns.tolist()}"
    )
    mask = func(adata.obs[col])
    return adata[mask]


def _recompute_expression_matrix(
    sdata,
    points_key: str = "transcripts",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int | None = "UNASSIGNED",
):
    transcripts = sdata.points[points_key].compute()

    # remove background transcripts
    transcripts = transcripts[~_is_background(transcripts[points_cell_id_key], points_background_id)]

    # Pivot: rows = cell IDs, columns = genes, values = counts
    expression_matrix_from_transcripts = (
        transcripts.groupby([points_cell_id_key, points_gene_key], observed=True).size().unstack(fill_value=0)
    )

    # Align the new expression matrix with the existing one in tables
    adata = sdata.tables[tables_key]
    genes = _get_genes(adata, tables_gene_key)
    # Ensure the new matrix has the same index and columns as the existing one
    expression_matrix_from_transcripts = expression_matrix_from_transcripts.reindex(
        index=adata.obs[tables_cell_id_key],
        columns=genes,
        fill_value=0,
    )
    return expression_matrix_from_transcripts


def _filter_control_and_low_quality_transcripts(
    sdata,
    min_qv: float | None = 20.0,
    control_genes: tuple | list = (),
    control_prefixes: tuple | list = (
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "DeprecatedCodeword_",
        "UnassignedCodeword_",
    ),
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_background_id: str | int | None = "UNASSIGNED",
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_gene_key: str | None = None,
    recompute_expression: bool = True,
    inplace: bool = True,
) -> sd.SpatialData:
    """
    Filter control and low-quality transcripts from the SpatialData object.
    This is always done in place.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing transcript data.
    min_qv : float | None, default=20.0
        Minimum quality value (qv) threshold for transcripts to be considered valid.
        If None, no filtering is applied based on quality.
    control_prefixes : tuple | list, default=(
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "DeprecatedCodeword_",
        "UnassignedCodeword_",
    )
        Control prefixes to identify control probes in gene names.
        Transcripts with gene names starting with any of these prefixes will be considered
        control probes and filtered out.
    control_genes : tuple | list, default=()
        Additional keywords to identify control probes in gene names.
        For these ones, exact matches will be filtered out (e.g. "GAPDH" or "ERCC-00002"),
        whereas for the control_prefixes, any gene name starting with the prefix will be
        filtered out (e.g. "NegControlProbe_1" or "NegControlProbe_2").
    points_key : str, default="transcripts"
        The key in the SpatialData points attribute that contains transcript data.
    points_gene_key : str, default="feature_name"
        The column name in the points DataFrame that contains gene names.
    points_cell_id_key : str, default="cell_id"
        The column name in the points DataFrame that contains cell IDs.
    points_background_id : str | int | None, default="UNASSIGNED"
        The value in the points DataFrame that indicates background/unassigned transcripts.
    tables_key : str, default="table"
        The key in the SpatialData tables attribute that contains the expression table.
    tables_cell_id_key : str, default="cell_id"
        The column name in the tables DataFrame that contains cell IDs.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    recompute_expression : bool, default=True
        Whether to recompute the expression matrix after filtering.
        Note that this can be computationally expensive for large datasets.
    inplace: bool, default=True
        Whether to modify the SpatialData object in place. Defaults to True.

    Returns
    -------
    sd.SpatialData
        The updated SpatialData object with invalid transcripts marked (in an extra column).
    """
    if inplace:
        warnings.warn(
            "Filtering control and low-quality transcripts from the SpatialData object in-place. "
            "Set filter_kwargs={'inplace': False} to avoid modifying the original object.",
            UserWarning,
            stacklevel=2,
        )
    else:
        sdata = sd.deepcopy(sdata)

    pts = sdata.points[points_key]
    adata = sdata.tables[tables_key]

    # materialize the df to perform the filtering
    pts_pd = pts.compute()
    # get the transformation
    points_transformation = pts.attrs["transform"]

    # we need multiple masks here:
    # one for the prefixed that checks using startswith
    # one for the genes that performs an exact match
    # one for the quality filtering (if qv column is present and min_qv is not None)
    prefix_mask = (
        pts_pd[points_gene_key].str.startswith(tuple(control_prefixes))
        if control_prefixes
        else pd.Series(False, index=pts_pd.index)
    )
    gene_mask = pts_pd[points_gene_key].isin(control_genes) if control_genes else pd.Series(False, index=pts_pd.index)

    if "qv" not in pts_pd.columns and min_qv is not None:
        raise KeyError(
            f"Quality value column 'qv' not found in points DataFrame. "
            f"Available columns: {pts.columns.tolist()}. "
            f"If you do not want to filter by quality, set min_qv=None."
        )
    elif "qv" not in pts_pd.columns and min_qv is None:
        invalid_mask = prefix_mask | gene_mask
    else:
        invalid_mask = prefix_mask | gene_mask | (pts_pd["qv"] < min_qv)

    removed_genes = pts_pd.loc[invalid_mask, points_gene_key].unique().tolist()
    pts_pd = pts_pd[~invalid_mask]
    sdata.points[points_key] = sd.models.PointsModel.parse(
        dd.from_pandas(pts_pd, npartitions=1), transformations=points_transformation
    )

    # ---- tables ----
    # on the anndata object, we remove genes that are control genes
    # filtering by quality does not make sense here, as we do not have per-gene quality values
    # again, we need to make a distinction between control prefixes (prefix match) and gene masks (exact match)
    genes_names = _get_genes(adata, tables_gene_key)
    prefix_mask = (
        genes_names.str.startswith(tuple(control_prefixes)) if control_prefixes else pd.Series(False, index=genes_names)
    )
    gene_mask = genes_names.isin(control_genes) if control_genes else pd.Series(False, index=genes_names)
    adata = adata[:, ~(prefix_mask | gene_mask)]
    sdata.tables[tables_key] = adata

    # check if any of the gene names of the removed transcripts appear in the anndata object
    # if so, that means we might need to recompute the expression matrix
    filtered_genes_in_adata = set(removed_genes) & set(genes_names)
    if len(filtered_genes_in_adata) > 0:
        if not recompute_expression:
            warnings.warn(
                f"Some of the filtered genes ({len(filtered_genes_in_adata)}) also appear in the tables. "
                f"These genes are: {list(filtered_genes_in_adata)[:5]}... "
                f"If you wish to recompute the expression matrix after filtering, set recompute_expression=True.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # aggregate the counts from the points to get a new expression matrix
            # the aggregate function from spatialdata is not sufficient,
            # because it removes all layers but the shapes and transcripts
            expression_matrix = _recompute_expression_matrix(
                sdata,
                points_key=points_key,
                tables_key=tables_key,
                tables_cell_id_key=tables_cell_id_key,
                points_gene_key=points_gene_key,
                points_cell_id_key=points_cell_id_key,
                points_background_id=points_background_id,
            )

            # updating the expression matrix in the tables
            adata = sdata.tables[tables_key].copy()
            # turn back into a sparse matrix
            adata.X = sp.csr_matrix(expression_matrix.values)
            sdata.tables[tables_key] = adata

    return sdata


## code written by claude.ai
def estimate_theta_simple(x):
    """
    Rough theta estimate from variance-to-mean relationship
    assuming var = mean + mean²/theta of a negative binomial distribution

    Parameters
    ----------
    x : numpy.ndarray
        The matrix containing the counts

    Returns
    -------
    float
        An estimate of the overdispersion parameter
    """

    gene_means = x.mean(axis=0)
    gene_vars = x.var(axis=0)

    # Only use genes with sufficient expression
    mask = (gene_means > 0.05) & (gene_vars > gene_means)

    # Solve: var = mean + mean²/theta for theta
    theta_estimates = gene_means[mask] ** 2 / (gene_vars[mask] - gene_means[mask])

    # Take median to be robust
    theta = np.median(theta_estimates[theta_estimates > 0])
    return theta


## adapted from https://github.com/scverse/scanpy licensed under BSD-3 to scverse
## implementing the method from Lause et al. (2021) https://link.springer.com/article/10.1186/s13059-021-02451-7
def pearson_residuals(x: np.ndarray, theta, clip: None):
    """
    Computes the Analytic pearson residuals from a negative binomial distribution to
    normalize the data

    Args:
        x (np.ndarray): The raw counts
        theta (float): The estimated overdispersion parameter
        clip: Whether or not to clip the variance, if None np.sqrt(n) is the max variance
        .

    Returns:
        pd.Series (bool): A boolean Series (True if background, False otherwise).
    """
    x = x.copy() if copy else x

    # check theta
    if theta <= 0:
        # TODO: would "underdispersion" with negative theta make sense?
        # then only theta=0 were undefined..
        msg = "Pearson residuals require theta > 0"
        raise ValueError(msg)
    # prepare clipping
    if clip is None:
        n = x.shape[0]
        clip = np.sqrt(n)
    if clip < 0:
        msg = "Pearson residuals require `clip>=0` or `clip=None`."
        raise ValueError(msg)

    sums_genes = np.sum(x, axis=0, keepdims=True)
    sums_cells = np.sum(x, axis=1, keepdims=True)
    sum_total = np.sum(sums_genes)

    mu = np.array(sums_cells @ sums_genes / sum_total)
    diff = np.array(x - mu)
    residuals = diff / np.sqrt(mu + mu**2 / theta)

    # clip
    residuals = np.clip(residuals, a_min=-clip, a_max=clip)

    # fill NA
    residuals = np.nan_to_num(residuals, nan=0.0)

    return residuals
