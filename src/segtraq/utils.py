# SegTraq/src/segtraq/utils/label_transfer.py

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.spatial.distance import cdist

from .bl import baseline as bl


def _to_ndarray(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _looks_like_counts(x, n: int = 1000, tol: float = 1e-8) -> bool:
    """Quickly check if data looks like non-negative integer counts."""
    arr = x.data if hasattr(x, "data") else np.asarray(x).ravel()
    if arr.size == 0:
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True
    samp = arr if arr.size <= n else np.random.choice(arr, n, replace=False)
    return np.all(samp >= 0) and np.allclose(samp, np.round(samp), atol=tol)


def assign_celltype_by_pearson(adata: AnnData, ref_mean_df: pd.DataFrame, cell_id_key: str = "cell_id") -> pd.DataFrame:
    """
    Assign cell types to cells in `adata` via Pearson correlation with reference means.

    Parameters
    ----------
    adata : AnnData
        Query dataset (log-normalized) with genes in `adata.var_names`.
    ref_mean_df : pd.DataFrame
        Reference matrix (cell_types x genes), log-normalized.
    cell_id_key: str
        Column name in tables DataFrame indicating cell IDs. Default is "cell_id".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['cell_id', 'celltype', 'pearson_corr'].
    """
    X_query = pd.DataFrame(
        _to_ndarray(adata.X),
        index=adata.obs[cell_id_key],
        columns=adata.var_names,
    )

    common_genes = X_query.columns.intersection(ref_mean_df.columns)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between query and reference.")

    X_query = X_query[common_genes]
    X_ref = ref_mean_df[common_genes]

    # correlation distance = 1 - Pearson correlation
    cor_mat = 1.0 - cdist(X_query.values, X_ref.values, metric="correlation")
    cor_df = pd.DataFrame(cor_mat, index=X_query.index, columns=X_ref.index)

    best_celltype = cor_df.idxmax(axis=1)
    best_score = cor_df.max(axis=1)

    return pd.DataFrame({"cell_id": X_query.index, "celltype": best_celltype.values, "pearson_corr": best_score.values})


def run_label_transfer(
    sdata,
    adata_reference: AnnData,
    celltype_key: str,
    table_key: str = "table",
    tx_min: float = 10.0,
    tx_max: float = 2000.0,
    gn_min: float = 5.0,
    gn_max: float = np.inf,
    label_key: str = "transferred_celltype",
    score_key: str = "transferred_celltype_corr",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Transfer cell labels from a reference AnnData to `sdata.tables[table_key]` by
    Pearson correlation to reference mean profiles.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[table_key]` as AnnData, and points needed for QC if absent.
        `sdata.tables[table_key].X` values are ideally normalized and log1p transformed.
        Otherwise transformation will be performed before running label transfer.
    adata_ref : AnnData
        Reference dataset (ideally normalized & log1p).
        Otherwise transformation will be performed before running label transfer.
    celltype_key : str
        Column in `adata_ref.obs` with reference cell types.
    table_key : str
        Key of the AnnData table in `sdata.tables`.
    tx_min, tx_max : float
        Min/max transcripts per cell for QC.
    gn_min, gn_max : float
        Min/max genes per cell for QC.
    label_key : str
        Column name to store transferred labels in `.obs` when `inplace=True`.
    score_key : str
        Column name to store correlation scores in `.obs` when `inplace=True`.
    inplace : bool
        If True, writes labels into `sdata.tables[table_key].obs` and returns None.
        If False, returns a DataFrame with ['cell_id', 'celltype', 'pearson_corr'].

    Returns
    -------
    None or pd.DataFrame
        None when `inplace=True`; otherwise a DataFrame of assignments.
    """

    if celltype_key not in adata_reference.obs.columns:
        raise KeyError(f"'{celltype_key}' not found in adata_ref.obs.")

    adata_ref = adata_reference.copy()

    if _looks_like_counts(adata_ref.X):
        warnings.warn(
            "Reference adata_ref does not appear log-normalized."
            "Counts will be log1p-transformed before running label transfer.",
            RuntimeWarning,
            stacklevel=2,
        )
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)

    counts = _to_ndarray(adata_ref.X)
    celltypes = adata_ref.obs[celltype_key]
    counts_df = pd.DataFrame(counts, columns=adata_ref.var_names)
    counts_df["celltype"] = celltypes.values
    ref_mean_df = counts_df.groupby("celltype").mean()

    tbl = sdata.tables[table_key]
    # Ensure QC columns exist; compute if missing
    need_tx = "transcript_count" not in tbl.obs.columns
    need_gn = "gene_count" not in tbl.obs.columns

    if need_tx or need_gn:
        tpc = bl.transcripts_per_cell(sdata).set_index("cell_id")
        gpc = bl.genes_per_cell(sdata).set_index("cell_id").rename(columns={"n_unique_genes": "gene_count"})
        to_join = gpc.join(tpc, how="outer")
        tbl.obs = tbl.obs.merge(
            to_join,
            how="left",
            left_on="cell_id",
            right_on="cell_id",
        ).reset_index(drop=True)

    # QC filter
    qc_range = {"transcript_count": (tx_min, tx_max), "gene_count": (gn_min, gn_max)}
    mask = np.ones(tbl.n_obs, dtype=bool)
    for key, (low, high) in qc_range.items():
        if key not in tbl.obs.columns:
            raise KeyError(f"QC column '{key}' not found in table.obs.")
        mask &= (tbl.obs[key].to_numpy() >= low) & (tbl.obs[key].to_numpy() <= high)

    # subset query AnnData and keep a copy for processing
    adata_q = tbl[mask].copy()

    # Ensure obs_names are cell ids (needed for merges/returns)
    # if "cell_id" in adata_q.obs.columns:
    #    adata_q.obs_names = adata_q.obs["cell_id"].astype(str)

    # Normalize & log1p (query)
    if _looks_like_counts(tbl.X):
        warnings.warn(
            "Spatialdata table appears to contain raw counts. "
            "Counts will be log1p-transformed before running label transfer.",
            RuntimeWarning,
            stacklevel=2,
        )
        sc.pp.normalize_total(adata_q, target_sum=1e4)
        sc.pp.log1p(adata_q)

    # Assign labels
    ct_corr = assign_celltype_by_pearson(adata_q, ref_mean_df)

    if inplace:
        # Write back only to the filtered subset cells
        out = ct_corr.rename(columns={"celltype": label_key, "pearson_corr": score_key})
        tbl.obs = tbl.obs.merge(out, how="left", left_on="cell_id", right_on="cell_id")
        tbl.obs[label_key] = tbl.obs[label_key].astype("category")
        return None
    else:
        return ct_corr
