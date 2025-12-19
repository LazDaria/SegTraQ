from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import warnings
from scipy import sparse
from tqdm.auto import tqdm

from ..utils import _score_one_list, _score_negative_with_neighbors, merge_into_obs, _looks_like_counts

from typing import Dict, Tuple, List
import numpy as np
from scipy.stats import fisher_exact

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_MECR(
    sdata,
    me_pairs_df: pd.DataFrame,
    cell_type_key: str,
    tables_key: str = "table",
    neighbors_key: str = "spatial_connectivities",
    use_raw: bool = True,
    obs_key_score: str = "MECR_contam_score",
    obs_key_n_pairs: str = "MECR_n_pairs",
    obs_key_n_coexpr: str = "MECR_n_coexpr",
) -> pd.DataFrame:
    """
    Compute a per-cell contamination score based on mutually exclusive marker pairs.

    For each spatial cell i:
      - take its cell type c_i,
      - find all neighbor cells via `neighbors_key`,
      - collect the neighbor types B(i),
      - look up all mutually exclusive marker pairs (gene_a, gene_b) for each
        (ct_a=c_i, ct_b in B(i)) in `me_pairs_df`,
      - count how many of these (gene_a, gene_b) pairs are co-expressed in cell i
        (both genes > 0),
      - define:
            score_i = n_coexpressed_pairs / n_relevant_pairs
        where n_relevant_pairs is the number of unique ME pairs relevant for
        the neighborhood of cell i.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[tables_key]` as AnnData and a neighbor graph
        in `.tables[tables_key].obsp[neighbors_key]`.
    me_pairs_df : pd.DataFrame
        DataFrame with mutually exclusive marker pairs, with at least columns:
           - "ct_a": focal cell type,
           - "ct_b": neighbor cell type,
           - "gene_a": marker gene of ct_a,
           - "gene_b": marker gene of ct_b.
    cell_type_key : str
        Column in `sdata.tables[tables_key].obs` with cell-type labels.
    tables_key : str, optional (default: "table")
        Key of the AnnData table in `sdata.tables`.
    neighbors_key : str, optional (default: "spatial_connectivities")
        Key in `adata.obsp` containing the cell-cell adjacency / connectivity
        matrix.
    use_raw : bool, optional (default: True)
        If True, use `.layers["raw"]` if present, otherwise `.X`.
    obs_key_score : str, optional
        Name of the `.obs` column where the contamination score is stored.
    obs_key_n_pairs : str, optional
        Name of the `.obs` column storing the number of relevant ME pairs per cell.
    obs_key_n_coexpr : str, optional
        Name of the `.obs` column storing the number of co-expressed ME pairs
        per cell.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with one row per cell containing:
            - "cell_id": index of the cell (adata.obs_names),
            - "cell_type",
            - obs_key_score,
            - obs_key_n_pairs,
            - obs_key_n_coexpr.
        The same information is also written into `adata.obs` under the
        specified column names.
    """
    adata = sdata.tables[tables_key]

    # --- 1) choose expression matrix ----------------------------------------
    if use_raw and "raw" in adata.layers:
        arr = adata.layers["raw"]
    else:
        arr = adata.X

    arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
    n_cells, n_genes = arr.shape

    # --- 2) indexing structures ---------------------------------------------
    var_index = pd.Index(adata.var_names)
    gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(var_index)}

    ctypes = adata.obs[cell_type_key].astype("category")
    ct_values = ctypes.values

    # adjacency matrix
    if neighbors_key not in adata.obsp:
        raise KeyError(f"Neighbors matrix '{neighbors_key}' not found in adata.obsp")

    G = adata.obsp[neighbors_key]  # expected CSR/CSC sparse matrix

    # --- 3) preprocess mutually exclusive pairs into index by (ct_a, ct_b) ---
    # keep only pairs where both genes are present in the spatial data
    valid_rows = me_pairs_df[
        me_pairs_df["gene_a"].isin(var_index)
        & me_pairs_df["gene_b"].isin(var_index)
    ].copy()

    pair_dict: Dict[Tuple[str, str], list[Tuple[int, int]]] = {}

    for row in valid_rows.itertuples(index=False):
        ct_a = row.ct_a
        ct_b = row.ct_b
        g_a = row.gene_a
        g_b = row.gene_b

        idx_a = gene_to_idx[g_a]
        idx_b = gene_to_idx[g_b]

        key = (ct_a, ct_b)
        pair_dict.setdefault(key, []).append((idx_a, idx_b))

    # make pairs unique per (ct_a, ct_b)
    for key, lst in pair_dict.items():
        pair_dict[key] = list(set(lst))

    # --- 4) per-cell contamination computation ------------------------------
    n_pairs_per_cell = np.zeros(n_cells, dtype=int)
    n_coexpr_per_cell = np.zeros(n_cells, dtype=int)

    for i in range(n_cells):
        ct_a = ct_values[i]

        # neighbors of cell i
        row = G.getrow(i)
        if row.nnz == 0:
            continue

        neighbor_indices = row.indices
        neighbor_types = pd.unique(ctypes.iloc[neighbor_indices].dropna())

        # collect all ME pairs relevant for this cell's neighborhood
        relevant_pairs: set[Tuple[int, int]] = set()
        for ct_b in neighbor_types:
            key = (ct_a, ct_b)
            if key in pair_dict:
                relevant_pairs.update(pair_dict[key])

        if not relevant_pairs:
            continue

        relevant_pairs = list(relevant_pairs)
        n_pairs_per_cell[i] = len(relevant_pairs)

        expr_i = arr[i, :]
        e_i = expr_i > 0

        coexpr_count = 0
        for idx_a, idx_b in relevant_pairs:
            if e_i[idx_a] and e_i[idx_b]:
                coexpr_count += 1

        n_coexpr_per_cell[i] = coexpr_count

    # contamination score = co-expressed / total pairs
    score = np.full(n_cells, np.nan, dtype=float)
    mask_pairs = n_pairs_per_cell > 0
    score[mask_pairs] = n_coexpr_per_cell[mask_pairs] / n_pairs_per_cell[mask_pairs]

    # --- 5) store in .obs and return a DataFrame ----------------------------
    adata.obs[obs_key_score] = score
    adata.obs[obs_key_n_pairs] = n_pairs_per_cell
    adata.obs[obs_key_n_coexpr] = n_coexpr_per_cell

    result_df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.to_numpy(),
            "cell_type": ct_values,
            obs_key_score: score,
            obs_key_n_pairs: n_pairs_per_cell,
            obs_key_n_coexpr: n_coexpr_per_cell,
        }
    ).set_index("cell_id")

    return result_df


def compute_MECR_uns(
    sdata,
    me_pairs_df: pd.DataFrame,
    cell_type_key: str,
    tables_key: str = "table",
    neighbors_key: str = "spatial_connectivities",
    cell_centroid_x_key: str = "cell_centroid_x",
    cell_centroid_y_key: str = "cell_centroid_y",
    use_raw: bool = True,
    inplace: bool = True,
    uns_key: str = "MECR_neighbors",
) -> pd.DataFrame:
    """
    Compute MECR (Fisher's exact test for mutual exclusivity) for mutually
    exclusive marker pairs, restricted to A cells that neighbor B cells.

    For each row in `me_pairs_df` (ct_a, ct_b, gene_a, gene_b):

        1. Select spatial cells of type ct_a that have at least one neighbor
           of type ct_b (based on `neighbors_key` in .obsp).

        2. On this subset of ct_a cells, build a 2x2 contingency table for
           gene_a and gene_b:

                     gene_b>0   gene_b=0
             gene_a>0    n11        n10
             gene_a=0    n01        n00

        3. Run Fisher's exact test with alternative="less":

             H0: gene_a and gene_b are independent
             H1: co-occurrence is LESS than expected (mutual exclusivity)

    This quantifies how much the B marker (gene_b) appears in A cells that
    are directly adjacent to B cells, i.e. a contamination-like signal.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[tables_key]` as AnnData.
    me_pairs_df : pd.DataFrame
        DataFrame with at least the columns:
            - "ct_a"   : focal cell type (where MECR is computed)
            - "ct_b"   : neighbor cell type
            - "gene_a" : marker gene of ct_a
            - "gene_b" : marker gene of ct_b
        Typically derived from mutually exclusive marker pairs in scRNA-seq.
    cell_type_key : str
        Column in `sdata.tables[tables_key].obs` with cell type labels for
        the spatial cells.
    tables_key : str, optional (default: "table")
        Key of the AnnData table in `sdata.tables`.
    neighbors_key : str, optional (default: "spatial_connectivities")
        Key in `adata.obsp` containing the adjacency / connectivity matrix
        (e.g. produced by Squidpy / Scanpy).
    use_raw : bool, optional (default: True)
        If True, use `.layers["raw"]` if present, otherwise `.X`.
        If False, always use `.X`.
    inplace : bool, optional (default: True)
        If True, store the result DataFrame in
            sdata.tables[tables_key].uns[uns_key].
    uns_key : str, optional (default: "MECR_neighbors")
        Key in `.uns` under which the result DataFrame is stored if
        `inplace=True`.

    Returns
    -------
    result_df : pd.DataFrame
        One row per input pair with columns:
            ["ct_a", "ct_b", "gene_a", "gene_b",
             "odds_ratio", "pval",
             "n_cells", "n11", "n10", "n01", "n00"]
        Rows where genes are missing or no cells are available contain NaNs.
    """
    adata = sdata.tables[tables_key]
    adata.obsm["spatial"] = adata.obs[[cell_centroid_x_key, cell_centroid_y_key]].to_numpy()

    # 1. Build spatial graph (Delaunay triangulation) #TODO
    sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
    G = adata.obsp["spatial_connectivities"].tocsr()

    # --- 1) choose expression matrix (prefer raw if requested) --------------
    if use_raw and "raw" in adata.layers:
        arr = adata.layers["raw"]
    else:
        arr = adata.X

    arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
    n_cells, n_genes = arr.shape

    # --- 2) basic indexing structures ---------------------------------------
    var_index = pd.Index(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_index)}

    # cell types as numpy array
    ctypes = adata.obs[cell_type_key].astype("category")
    ct_values = ctypes.values
    ct_masks = {ct: (ct_values == ct) for ct in ctypes.cat.categories}

    # adjacency / neighbor matrix
    if neighbors_key not in adata.obsp:
        raise KeyError(f"Neighbors matrix '{neighbors_key}' not found in adata.obsp")

    G = adata.obsp[neighbors_key]  # expected CSR or similar sparse matrix

    # --- 3) precompute mask of "ct_a cells that have a ct_b neighbor" -------
    pair_masks = {}
    for ct_a, ct_b in (
        me_pairs_df[["ct_a", "ct_b"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        if ct_a not in ct_masks or ct_b not in ct_masks:
            pair_masks[(ct_a, ct_b)] = None
            continue

        mask_a = ct_masks[ct_a]
        mask_b = ct_masks[ct_b]

        # For each cell, count number of neighbors that are ct_b
        # G.shape = (n_cells, n_cells), mask_b.shape = (n_cells,)
        # -> neighbor_counts = G @ mask_b
        neighbor_counts = G.dot(mask_b.astype(int))
        has_b_neighbor = neighbor_counts > 0

        # only A cells that have at least one B neighbor
        subset_mask = mask_a & has_b_neighbor
        if subset_mask.sum() == 0:
            pair_masks[(ct_a, ct_b)] = None
        else:
            pair_masks[(ct_a, ct_b)] = subset_mask

    # --- 4) loop over gene pairs and run Fisher in the relevant subset ------
    records = []

    for row in me_pairs_df.itertuples(index=False):
        ct_a = row.ct_a
        ct_b = row.ct_b
        g1 = row.gene_a
        g2 = row.gene_b

        mask = pair_masks.get((ct_a, ct_b), None)
        if mask is None or mask.sum() == 0:
            records.append(
                dict(
                    ct_a=ct_a,
                    ct_b=ct_b,
                    gene_a=g1,
                    gene_b=g2,
                    odds_ratio=np.nan,
                    pval=np.nan,
                    n_cells=0,
                    n11=np.nan,
                    n10=np.nan,
                    n01=np.nan,
                    n00=np.nan,
                )
            )
            continue

        # restrict to ct_a cells with at least one ct_b neighbor
        sub_arr = arr[mask, :]
        n_sub = sub_arr.shape[0]

        # gene indices
        idx1 = gene_to_idx.get(g1)
        idx2 = gene_to_idx.get(g2)
        if idx1 is None or idx2 is None:
            records.append(
                dict(
                    ct_a=ct_a,
                    ct_b=ct_b,
                    gene_a=g1,
                    gene_b=g2,
                    odds_ratio=np.nan,
                    pval=np.nan,
                    n_cells=n_sub,
                    n11=np.nan,
                    n10=np.nan,
                    n01=np.nan,
                    n00=np.nan,
                )
            )
            continue

        expr1 = sub_arr[:, idx1]
        expr2 = sub_arr[:, idx2]

        e1 = expr1 > 0
        e2 = expr2 > 0

        # contingency table
        n11 = int((e1 & e2).sum())          # gene_a>0, gene_b>0
        n10 = int((e1 & ~e2).sum())         # gene_a>0, gene_b=0
        n01 = int((~e1 & e2).sum())         # gene_a=0, gene_b>0
        n00 = int(n_sub - n11 - n10 - n01)  # gene_a=0, gene_b=0

        # if (n11 + n10 + n01) == 0:
        #     odds_ratio, pval = np.nan, np.nan
        # else:
        #     table = [[n11, n10], [n01, n00]]
        #     try:
        #         odds_ratio, pval = fisher_exact(table, alternative="less")
        #     except Exception:
        #         odds_ratio, pval = np.nan, np.nan

        records.append(
            dict(
                ct_a=ct_a,
                ct_b=ct_b,
                gene_a=g1,
                gene_b=g2,
                # odds_ratio=odds_ratio,
                # pval=pval,
                n_cells=n_sub,
                n11=n11,
                n10=n10,
                n01=n01,
                n00=n00,
            )
        )

    result_df = pd.DataFrame.from_records(records)

    if inplace:
        adata.uns[uns_key] = result_df

    return result_df

def compute_MECR_global(
    sdata,
    gene_pairs: List[Tuple[str, str]],
    tables_key: str = "table",
    inplace: bool = True,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Compute Fisher's exact test for mutual exclusivity per gene pair.

    For each (gene1, gene2) pair, this function:
        - binarizes expression as > 0 (present/absent),
        - builds a 2x2 contingency table:

              B>0   B=0
          A>0  a     b
          A=0  c     d

        - runs Fisher's exact test with alternative="less", i.e.:

              H0: A and B are independent
              H1: A and B co-occur LESS than expected (mutual exclusivity)

        - returns the odds ratio and p-value for each pair.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[tables_key]` as AnnData.
    gene_pairs : list of tuple
        Collection of (gene1, gene2) pairs (e.g. from `segtraq.get_mut_excl_markers`).
    tables_key : str, optional (default: "table")
        Key of the AnnData table in `sdata.tables`.
    inplace : bool, optional (default: True)
        If True, store results in:
            sdata.tables[tables_key].uns["Fisher_OR"]
            sdata.tables[tables_key].uns["Fisher_pval"]

    Returns
    -------
    or_dict : dict
        Mapping {(gene1, gene2): odds_ratio}.
        Odds ratio < 1 suggests mutual exclusivity (given a small p-value).
    pval_dict : dict
        Mapping {(gene1, gene2): p_value} for the one-sided test
        (alternative="less"). Small p-values indicate significant
        under-co-occurrence (mutual exclusivity).
    """
    tbl = sdata.tables[tables_key]

    # --- 1) Choose a raw-count matrix ---------------------------------------
    X = tbl.X

    X = tbl.X
    # Check if X looks like counts
    if _looks_like_counts(X):
        arr = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in tbl.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = tbl.layers["raw"]
        arr = raw.toarray() if hasattr(raw, "toarray") else raw

    # --- 2) Set up gene indexing --------------------------------------------
    var_index = pd.Index(tbl.var_names)
    n_cells = arr.shape[0]

    or_dict: Dict[Tuple[str, str], float] = {}
    pval_dict: Dict[Tuple[str, str], float] = {}

    # --- 3) Loop over pairs and run Fisher ----------------------------------
    for g1, g2 in gene_pairs:
        if g1 not in var_index or g2 not in var_index:
            or_dict[(g1, g2)] = np.nan
            pval_dict[(g1, g2)] = np.nan
            continue

        idx1 = var_index.get_loc(g1)
        idx2 = var_index.get_loc(g2)

        expr1 = arr[:, idx1]
        expr2 = arr[:, idx2]

        # binarize raw counts: present vs absent
        e1 = expr1 > 0
        e2 = expr2 > 0

        # contingency table entries
        a = int((e1 & e2).sum())                 # A>0, B>0
        b = int((e1 & ~e2).sum())                # A>0, B=0
        c = int((~e1 & e2).sum())                # A=0, B>0
        d = n_cells - a - b - c                  # A=0, B=0

        # if no detections at all, nothing to test
        if (a + b + c) == 0:
            or_dict[(g1, g2)] = np.nan
            pval_dict[(g1, g2)] = np.nan
            continue

        table = [[a, b], [c, d]]

        try:
            odds_ratio, pval = fisher_exact(table, alternative="less")
        except Exception:
            odds_ratio, pval = np.nan, np.nan

        or_dict[(g1, g2)] = odds_ratio
        pval_dict[(g1, g2)] = pval

    # --- 4) Store in .uns if requested --------------------------------------
    if inplace:
        tbl.uns.setdefault("Fisher_OR", {}).update(or_dict)
        tbl.uns.setdefault("Fisher_pval", {}).update(pval_dict)

    return or_dict, pval_dict

def calculate_neighbor_contamination(
    sdata,
    cell_type_key: str,
    markers: Dict[str, Dict[str, List[str]]],
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    uns_key: str = "negative_marker_contamination",
    uns_key_binary: str = "negative_marker_contamination_binary",
    inplace: bool = True,
):
    """
    Compute per-cell negative-marker contamination and directed c_src→c_tgt
    contamination summaries.

    Per-cell outputs (written to .obs):
        - neg_marker_contam_counts:
            Total transcripts in the focal cell that belong to genes that are
            (i) negative markers of the focal cell type and
            (ii) positive markers of at least one neighboring cell type.
        - neg_marker_contam_fraction:
            For each such gene g, compute x_i(g) / (x_i(g) + mean_neighbor(g)),
            averaged across genes (neighbors pooled across all neighbor types).

    Type x type outputs (written to .uns):
        - uns_key:
            A directed matrix with entries = mean over genes of
            x_i(g) / (x_i(g) + mean_src(g)), aggregated across all contributing
            (cell, gene) observations for (c_src, c_tgt).
        - uns_key_binary:
            A directed matrix with entries =
            (# target cells of type c_tgt contaminated by c_src) / (# target cells of type c_tgt).
            Here “contaminated by c_src” is binary per target cell: at least one
            relevant gene (negative in target, positive in source) is detected in the
            target cell and detected in at least one neighbor of type c_src.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is 
        expressed in at least one neighboring cell of the source type.
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    uns_key : str, optional, default="negative_marker_contamination"
        Key in `.uns` under which the directed source → target mean contamination
        fraction matrix is stored.
    uns_key_binary : str, optional, default="negative_marker_contamination_binary"
        Key in `.uns` under which the directed source → target binary contamination
        proportion matrix is stored.
    inplace : bool, optional, default=True
        If True, store per-cell contamination metrics in
        `sdata.tables[tables_key].obs` and type-level matrices in `.uns`.

    Returns
    -------
    per_cell_df : pd.DataFrame
        Per-cell contamination metrics, indexed by cell ID.
    contam_matrix_df : pd.DataFrame
        Directed type x type mean contamination fraction matrix (c_src rows, c_tgt columns).
    contam_binary_df : pd.DataFrame
        Directed type x type binary contamination proportion matrix (c_src rows, c_tgt columns).
    """

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    adata = sdata.tables[tables_key]
    X = adata.X
    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X.shape[0]

    # Dense expression (counts)
    if _looks_like_counts(X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in adata.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = adata.layers["raw"]
        X_dense = raw.toarray() if hasattr(raw, "toarray") else raw

    # Neighbors
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key={neighbors_key} missing; computing Delaunay neighbors.",
            RuntimeWarning,
        )
        adata.obsm["spatial"] = adata.obs[
            [tables_centroid_x_key, tables_centroid_y_key]
        ].to_numpy()
        import squidpy as sq  # local import so this function doesn't hard-require squidpy
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Marker sets
    pos_sets = {ct: set(m.get("positive", [])) for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) for ct, m in markers.items()}

    # all cell types actually present
    all_cts = sorted({ct for ct in cell_types if not pd.isna(ct)})

    # denominator per target type for binary matrix
    tgt_totals = {ct: int(np.sum(cell_types == ct)) for ct in all_cts}  

    # ----------------------------------------------------------------------
    # Precompute for each (c_src, c_tgt): negative(c_tgt) ∩ positive(c_src)
    # ----------------------------------------------------------------------
    type_pair_genes: Dict[Tuple[str, str], np.ndarray] = {}

    for c_tgt in all_cts:
        neg = neg_sets.get(c_tgt, set())
        if not neg:
            continue
        for c_src in all_cts:
            pos = pos_sets.get(c_src, set())
            genes_inter = list(neg & pos)
            if not genes_inter:
                continue
            idx = var_index.get_indexer(genes_inter)
            idx = idx[idx >= 0]
            if idx.size:
                type_pair_genes[(c_src, c_tgt)] = idx

    # ----------------------------------------------------------------------
    # Accumulators
    # ----------------------------------------------------------------------
    numer_cell = np.zeros(n_cells, dtype=float)
    sum_cell_frac = np.zeros(n_cells, dtype=float)
    count_cell_genes = np.zeros(n_cells, dtype=int)

    sum_pair = defaultdict(float)   # mean fraction numerator per (c_src, c_tgt)
    count_pair = defaultdict(int)   # gene contributions per (c_src, c_tgt)

    contam_cells_hit = defaultdict(int)  #target cells hit by source (binary)

    # ----------------------------------------------------------------------
    # Loop over cells
    # ----------------------------------------------------------------------
    for i in range(n_cells):
        c_tgt = cell_types[i]
        if pd.isna(c_tgt) or c_tgt not in neg_sets:
            continue

        nbs = neighbor_indices[i]
        if len(nbs) == 0:
            continue

        x_i = X_dense[i, :]
        nb_cts = cell_types[nbs]

        # track per-cell genes already counted (per-cell metrics)
        used_genes = set()

        # track which source types contaminate this target cell at least once
        contaminated_by_src = set()  

        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            if pair not in type_pair_genes:
                continue

            gene_idx = type_pair_genes[pair]
            x_i_sub = x_i[gene_idx]

            # neighbors of this source type
            nb_src = nbs[nb_cts == c_src]
            if len(nb_src) == 0:
                continue

            X_nb_src = X_dense[np.ix_(nb_src, gene_idx)]

            # loop genes
            for k, g_idx in enumerate(gene_idx):
                x_i_g = x_i_sub[k]
                if x_i_g <= 0:
                    continue

                #only call it contamination if at least one neighbor expresses the gene
                x_nb_g = X_nb_src[:, k]
                if require_neighbor_expression:
                    mask_pos = x_nb_g > 0
                    if not mask_pos.any():
                        continue
                    mean_src = x_nb_g[mask_pos].mean()
                else:
                    mean_src = x_nb_g.mean()

                if not np.isfinite(mean_src):
                    continue

                # binary “this target cell is contaminated by this source type”
                contaminated_by_src.add(c_src) 

                
                denom_all = x_i_g + mean_src
                if denom_all <= 0:
                    continue
                frac_g = x_i_g / denom_all

                # pair stats: always update (no used_genes gating)
                sum_pair[pair] += frac_g
                count_pair[pair] += 1

                # per-cell stats: update once per gene per cell
                if g_idx not in used_genes:
                    numer_cell[i] += x_i_g
                    sum_cell_frac[i] += frac_g
                    count_cell_genes[i] += 1
                    used_genes.add(g_idx)

        # update per-(c_src, c_tgt) binary hit counts once per cell
        for c_src in contaminated_by_src:            
            contam_cells_hit[(c_src, c_tgt)] += 1     

    # ----------------------------------------------------------------------
    # Build per-cell output
    # ----------------------------------------------------------------------
    contam_fraction = np.divide(
        sum_cell_frac,
        count_cell_genes,
        out=np.full(n_cells, np.nan),
        where=count_cell_genes > 0,
    )

    per_cell_df = pd.DataFrame(
        {
            "neg_marker_contam_counts": numer_cell,
            "neg_marker_contam_fraction": contam_fraction,
        },
        index=adata.obs[tables_cell_id_key],
    )

    # ----------------------------------------------------------------------
    # Build contamination matrices contam_matrix_df, contam_binary_df
    # ----------------------------------------------------------------------
    ct_list = all_cts
    idxmap = {ct: j for j, ct in enumerate(ct_list)}

    contam_mat = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)
    for (c_src, c_tgt), total in sum_pair.items():
        n = count_pair[(c_src, c_tgt)]
        if n > 0:
            contam_mat[idxmap[c_src], idxmap[c_tgt]] = total / n

    contam_matrix_df = pd.DataFrame(contam_mat, index=ct_list, columns=ct_list)

    # binary target-normalized matrix
    contam_bin = np.full((len(ct_list), len(ct_list)), np.nan, dtype=float)  
    for (c_src, c_tgt), hits in contam_cells_hit.items():                   
        denom = tgt_totals.get(c_tgt, 0)                                    
        if denom > 0:                                                    
            contam_bin[idxmap[c_src], idxmap[c_tgt]] = hits / denom          
    contam_binary_df = pd.DataFrame(contam_bin, index=ct_list, columns=ct_list) 

    # ----------------------------------------------------------------------
    # Write to .obs and .uns
    # ----------------------------------------------------------------------
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        sdata.tables[tables_key].uns[uns_key] = contam_matrix_df
        sdata.tables[tables_key].uns[uns_key_binary] = contam_binary_df  

    return per_cell_df, contam_matrix_df, contam_binary_df

def calculate_marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    use_quantiles: bool = False,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    weight_cont: float = 0.7,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using positive markers globally
    and negative markers restricted to the focal cell's neighborhood.

    Positive markers:
        - For each cell's annotated type c, use markers[c]["positive"].
        - Compute per-cell precision/recall/F1 relative to these markers,
          using either a quantile-based threshold or expression > 0.

    Negative markers (neighborhood-aware):
        - For a cell i of type c:
            * Find the cell types present in its spatial neighborhood,
              using `adata.obsp[neighbors_key]`.
            * Take c's negative markers: markers[c]["negative"].
            * Intersect them with the union of positive markers of the
              neighbor cell types.
            * Consider only genes that are present in at least one neighbor
              of the source type (if `require_neighbor_expression=True`).
            → "relevant negatives" for this cell.
        - Compute per-cell precision/recall/F1 for these relevant negatives.

    Finally:
        - Combine positive_F1 and negative_F1 into an overall F1_purity
          that rewards high positive-F1 and low negative-F1. Weighting 
          factor weight_cont controls the importance of negative-F1.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}.
    use_quantiles : bool, optional, default=False
        If True, define predictions by the top-|markers| fraction per cell (rank-based);
        if False, use direct expression-based criteria (expression > 0).
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is 
        expressed in at least one neighboring cell of the source type.
    weight_cont : float, optional, default=0.7
        Weighting factor for negative marker F1 in the overall F1_purity.
        Higher values give more weight to negative F1. Must be in the range [0, 1].
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            [
             'positive_precision', 'positive_recall', 'positive_F1',
             'negative_precision', 'negative_recall', 'negative_F1',
             'F1_purity']
        indexed by cell ID.
    """
    if not (0 <= weight_cont <= 1):
        raise ValueError(
            f"weight_cont must be between 0 and 1 (inclusive), got {weight_cont}"
        )

    adata = sdata.tables[tables_key]

    X = adata.X  # keep sparse if sparse
    
    if _looks_like_counts(X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
    elif "raw" not in adata.layers:
        raise ValueError(
            f"'raw' layer does not exist in sdata.tables['{tables_key}'], "
            "and the main matrix does not look like counts."
        )
    else:
        raw = adata.layers["raw"]
        X_dense = raw.toarray() if hasattr(raw, "toarray") else raw

    genes = np.asarray(adata.var_names)
    var_index = pd.Index(genes)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells, _n_genes = X.shape

    def _idx(lst) -> np.ndarray:
        if lst is None or len(lst) == 0:
            return np.empty(0, dtype=int)
        return var_index.get_indexer_for(lst)

    # Output arrays
    pos_prec = np.full(n_cells, np.nan, dtype=float)
    pos_rec  = np.full(n_cells, np.nan, dtype=float)
    pos_f1   = np.full(n_cells, np.nan, dtype=float)

    neg_prec = np.full(n_cells, np.nan, dtype=float)
    neg_rec  = np.full(n_cells, np.nan, dtype=float)
    neg_f1   = np.full(n_cells, np.nan, dtype=float)

    purity   = np.full(n_cells, np.nan, dtype=float)

    # ---------------- POSITIVE markers (per type, global) -------------------
    unique_cts = pd.unique(cell_types)

    for ct in unique_cts:
        mask_cells = (cell_types == ct)
        idx_cells = np.where(mask_cells)[0]
        if idx_cells.size == 0:
            continue

        # missing/unknown type → leave NaNs
        if pd.isna(ct) or ct not in markers:
            continue

        m = markers[ct]
        pos_idx = _idx(m.get("positive", []))
        neg_idx = _idx(m.get("negative", []))
        pos_idx = pos_idx[pos_idx >= 0]
        neg_idx = neg_idx[neg_idx >= 0]
        all_idx = np.unique(np.concatenate([pos_idx, neg_idx])) if neg_idx.size else pos_idx

        # If no pos markers, nothing to compute
        if pos_idx.size == 0:
            continue

        X_ct = X_dense[mask_cells]

        p_prec_ct, p_rec_ct, p_f1_ct = _score_one_list(
            X_ct,
            pos_idx,
            all_idx,
            use_quantiles=use_quantiles,
        )
        pos_prec[idx_cells] = p_prec_ct
        pos_rec[idx_cells]  = p_rec_ct
        pos_f1[idx_cells]   = p_f1_ct

    # ------------- NEGATIVE markers: neighborhood-aware ---------------------
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key='{neighbors_key}' not found in adata.obsp. "
            "A neighborhood graph based on Delaunay will be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        sdata[tables_key].obsm["spatial"] = sdata[tables_key].obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")

    G = adata.obsp[neighbors_key]

    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    neg_prec, neg_rec, neg_f1 = _score_negative_with_neighbors(
        X_dense=X_dense,
        cell_types=cell_types,
        markers=markers,
        genes=genes,
        require_neighbor_expression=require_neighbor_expression,
        neighbor_indices=neighbor_indices,
        use_quantiles=use_quantiles,
    )

    # ------------------------ F1_purity -------------------------------------
    p_f1_all = pos_f1
    n_f1_all = neg_f1

    mask_valid = ~np.isnan(p_f1_all) & ~np.isnan(n_f1_all)
    if np.any(mask_valid):
        p = p_f1_all[mask_valid]
        n = 1 - n_f1_all[mask_valid]
        weight_id = 1 - weight_cont
        purity_vals = weight_cont * n + weight_id * p
        purity[mask_valid] = purity_vals
    # cells without pos or neg markers stay NaN for purity

    # Build DataFrame
    result = pd.DataFrame(
        {
            "positive_precision": pos_prec,
            "positive_recall": pos_rec,
            "positive_F1": pos_f1,
            "negative_precision": neg_prec,
            "negative_recall": neg_rec,
            "negative_F1": neg_f1,
            "F1_purity": purity,
        },
        index=adata.obs[tables_cell_id_key],
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=result,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )

    return result
