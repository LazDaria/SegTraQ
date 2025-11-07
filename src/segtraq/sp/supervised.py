from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse
from tqdm.auto import tqdm

from ..utils import _score_one_list, merge_into_obs


def compute_MECR(
    sdata, gene_pairs: list[tuple[str, str]], tables_key: str = "table", inplace: bool = True
) -> dict[tuple[str, str], float]:
    """
    Modified from https://github.com/dpeerlab/segger-analysis

    Compute Mutually Exclusive Co-expression Rate (MECR) per gene pair.

    Parameters
    ----------
    sdata : SpatialData-like
        Container with `.tables[tables_key]` as AnnData.
    gene_pairs : list of tuple
        Collection of (gene1, gene2) pairs computed via `segtraq.get_mut_excl_markers`.
    tables_key : str
        Key of the AnnData table in `sdata.tables`.
    inplace : bool, optional
        If True, store MECR results in `sdata.tables['table'].uns['MECR']`.

    Returns
    -------
    dict
        Mapping {(gene1, gene2): MECR}, where MECR = P(both>0) / P(at least one>0).
    """
    mecr: dict[tuple[str, str], float] = {}
    expr_df = sdata.tables[tables_key].to_df()

    for g1, g2 in gene_pairs:
        e1 = expr_df[g1] > 0
        e2 = expr_df[g2] > 0
        p_both = (e1 & e2).mean()
        p_any = (e1 | e2).mean()
        mecr[(g1, g2)] = (p_both / p_any) if p_any > 0 else 0.0

    if inplace:
        if "MECR" not in sdata.tables[tables_key].uns:
            sdata.tables[tables_key].uns["MECR"] = {}
        sdata.tables[tables_key].uns["MECR"].update(mecr)

    return mecr


def calculate_contamination(
    sdata,
    markers,
    cell_type_key: str,
    tables_key: str = "table",
    radius: float = 15,
    n_neighs: int = 10,
    num_cells: int = 10_000,
    seed: int = 0,
    cell_centroid_x_key: str = "cell_centroid_x",
    cell_centroid_y_key: str = "cell_centroid_y",
    weight_edges: bool = False,
    inplace: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Modified from https://github.com/dpeerlab/segger-analysis

    Compute directional contamination (“leakage”) between cell types from spatial neighbors using
    ct-specific positive markers: for each ct cell and neighbor ct2, measure the fraction of ct-marker
    signal in ct2 neighbors relative to the ct neighborhood total, then average over interactions.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as AnnData with expression and coordinates.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str] (optional)}}, using only "positive".
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    tables_key : str, optional
        Key of the AnnData table in `sdata.tables`.
    radius : float, optional
        Radius for spatial neighbor construction.
    n_neighs : int, optional
        Max number of neighbors per cell (used with `radius`).
    num_cells : int, optional
        Number of cells to sample (speed/precision trade-off).
    seed : int, optional
        RNG seed for reproducible sampling.
    cell_centroid_x_key : str, optional
        `.obs` key for x-coordinates (used to build `.obsm["spatial"]` if needed).
    cell_centroid_y_key : str, optional
        `.obs` key for y-coordinates (used to build `.obsm["spatial"]` if needed).
    weight_edges : bool, optional
        Weight neighbor contributions by graph edge weights if True.
    inplace : bool, optional
        If True, store contamination matrix in `sdata.tables['table'].uns['contamination']`.

    Returns
    -------
    out: pandas.DataFrame
        Rows = source types (ct), columns = target types (ct2); entry is the mean fraction of
        ct-specific marker counts found in ct2 neighbors relative to ct neighborhood totals (directional).
    records_df: pandas.DataFrame
        Pandas dataframe with per-cell cell_id (ct), cell_type (ct), neighbor_id (ct2), neigbhor_type (ct2) and ratio
        (ct-specific markers counts found in ct2 neighbors relative to ct neighborhood total counts).
    """

    adata = sdata.tables[tables_key]

    adata.obsm["spatial"] = adata.obs[[cell_centroid_x_key, cell_centroid_y_key]].to_numpy()
    sq.gr.spatial_neighbors(adata, radius=radius, n_neighs=n_neighs, coord_type="generic")
    G = adata.obsp["spatial_connectivities"].tocsr()

    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X))
    else:
        X = X.tocsr()

    # library sizes
    # libsize = np.asarray(X.sum(axis=1)).ravel()
    # mean_lib = float(libsize.mean())

    var_index = pd.Index(adata.var_names)
    pos_markers = {ct: [g for g in set(m.get("positive", [])) if g in var_index] for ct, m in markers.items()}
    gene_idx = {g: var_index.get_loc(g) for gs in pos_markers.values() for g in gs}

    ct_list = list(pos_markers)
    diff_cols = {}
    for ct in ct_list:
        set_ct = set(pos_markers[ct])
        for ct2 in ct_list:
            if ct == ct2:
                continue
            cols = [gene_idx[g] for g in set_ct.difference(pos_markers[ct2])]
            diff_cols[(ct, ct2)] = np.array(cols, dtype=int)

    rng = np.random.default_rng(seed)
    n = adata.n_obs
    idx_cells = rng.choice(n, size=min(num_cells, n), replace=False)

    types = np.asarray(adata.obs[cell_type_key])
    C_sum = defaultdict(lambda: defaultdict(float))
    C_cnt = defaultdict(lambda: defaultdict(int))

    if weight_edges:
        # normalized weights can be derived from G; here we just use G as-is and add identity
        G_eff = G + sparse.eye(G.shape[0], format="csr")
    else:
        # unweighted: treat neighbors equally
        G_eff = None

    records = []

    for i in idx_cells:
        ct = types[i]

        start, end = G.indptr[i], G.indptr[i + 1]
        neigh = G.indices[start:end]
        if neigh.size == 0:
            continue

        for j in neigh:
            ct2 = types[j]
            if ct2 == ct:
                continue
            cols = diff_cols.get((ct, ct2))
            if cols is None or cols.size == 0:
                continue

            # numerator: neighbor j counts over S(ct, ct2) and scale by its total counts
            # TODO - check if this makes sense
            num = X[j, cols].sum()  # / (libsize[j] / mean_lib)

            # denominator: counts over S in (i ∪ N(i))
            if weight_edges:
                # weighted neighborhood sum: use (row i of G_eff) as weights
                w_idx = G_eff.indices[G_eff.indptr[i] : G_eff.indptr[i + 1]]
                w_val = G_eff.data[G_eff.indptr[i] : G_eff.indptr[i + 1]]
                # sum over rows w-weighted: (w^T @ X[:, cols]) -> use sparse vector-matrix product
                denom = (sparse.csr_matrix((w_val, ([0] * len(w_idx), w_idx)), shape=(1, n)) @ X[:, cols]).sum()
            else:
                # unweighted: sum rows {i} ∪ neigh
                denom = X[i, cols].sum() + X[neigh, :][:, cols].sum()

            denom = float(denom)
            if denom > 0.0:
                C_sum[ct][ct2] += float(num) / denom
                C_cnt[ct][ct2] += 1

                records.append(
                    {
                        "cell_id": int(i),
                        "cell_type": ct,
                        "neighbor_id": int(j),
                        "neighbor_type": ct2,
                        "ratio": float(num) / denom,
                    }
                )
                records_df = pd.DataFrame(records)

    cts = sorted(pos_markers)
    out = pd.DataFrame(0.0, index=cts, columns=cts)
    for ct in cts:
        for ct2 in cts:
            if ct == ct2:
                continue
            k = C_cnt[ct][ct2]
            out.loc[ct, ct2] = C_sum[ct][ct2] / k if k else 0.0

    out.index.name = "Source Cell Type"
    out.columns.name = "Target Cell Type"

    if inplace:
        if "contamination" not in adata.uns:
            adata.uns["contamination"] = out
        else:
            adata.uns["contamination"].update(out)

    return C_cnt, out, records_df  # delete C_cnt later!!!


def calculate_marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    use_quantiles: bool = True,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    # Translated and modified from https://github.com/SydneyBioX/CellSPA

    Compute per-cell marker purity: for each cell's annotated type, evaluate Precision/Recall/F1
    using its positive and negative marker lists, then summarize into an overall `F1_purity`
    that rewards high positive-F1 and low negative-F1.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict
        {cell_type: {"positive": list[str], "negative": list[str]}}; both lists are required.
    use_quantiles : bool, optional
        If True, define predictions by the top-|markers| fraction per cell (rank-based);
        if False, use direct expression-based criteria (e.g., >0).
    tables_key : str, optional
        Key of the AnnData table in `sdata.tables`.
    tables_cell_id_key : str, optional
        Column in the AnnData `.obs` with unique cell IDs.
    inplace : bool, optional
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns: ['positive_precision','positive_recall','positive_F1',
                'negative_precision','negative_recall','negative_F1',
                'F1_purity','cell_type'] indexed by cell.
    """
    adata = sdata.tables[tables_key]

    # dense view for quantiles; adjust if you need to stay sparse
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    genes = np.asarray(adata.var_names)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells, n_genes = X.shape

    def _idx(lst: list[str]) -> np.ndarray:
        if not lst:
            return np.empty(0, dtype=int)
        return np.where(np.isin(genes, np.asarray(lst)))[0]

    pos_idx_map = {ct: _idx(m.get("positive", [])) for ct, m in markers.items()}
    neg_idx_map = {ct: _idx(m.get("negative", [])) for ct, m in markers.items()}

    rows = []
    for i in range(n_cells):
        ct = cell_types[i]
        expr = X[i, :]

        # positive pass (upper quantile)
        p_prec, p_rec, p_f1 = _score_one_list(expr, pos_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # negative pass (upper quantile)
        n_prec, n_rec, n_f1 = _score_one_list(expr, neg_idx_map.get(ct, np.empty(0, dtype=int)), n_genes, use_quantiles)

        # fused purity
        denom = (1.0 - n_f1) + p_f1
        f1_purity = (2.0 * (1.0 - n_f1) * p_f1 / denom) if denom > 0 else 0.0

        rows.append(
            {
                "positive_precision": p_prec,
                "positive_recall": p_rec,
                "positive_F1": p_f1,
                "negative_precision": n_prec,
                "negative_recall": n_rec,
                "negative_F1": n_f1,
                "F1_purity": f1_purity,
            }
        )

    result = pd.DataFrame(rows, index=adata.obs[tables_cell_id_key])
    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=result,
            table_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
    return result


def calculate_diff_abundance(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]],
    tables_key: str = "table",
    lfc_thresh: float = 1.0,  # noqa
    pval_thresh: float = 0.05,  # noqa
    min_n_cells: int = 20,
    min_n_transcripts: int = 20,
    seed: int = 0,
    cell_centroid_x_key: str = "cell_centroid_x",
    cell_centroid_y_key: str = "cell_centroid_y",
    inplace: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate differential transcript abundance between bordering and non-bordering cells
    for every ordered pair of cell types using a spatial graph (Delaunay triangulation).
    This function builds a spatial neighbor graph from the provided AnnData table, classifies
    cells of each source cell type (ct1) into two groups depending on whether they border
    cells of a target cell type (ct2), and performs a differential abundance test (Scanpy's
    rank_genes_groups with Wilcoxon test) between the "bordering" and "non_bordering" groups.
    Results are filtered by transcript counts and by provided marker lists so that only genes
    likely to originate from spillover (positive markers of the source cell type ct2 and
    not negative markers of the receiver ct1) are retained. A summary table of significant
    genes per (ct1, ct2) pair is returned.

    Parameters
    ----------
    sdata : object
        Object containing AnnData tables, expected to expose a mapping-like attribute
        `tables` such that `sdata.tables[tables_key]` is an AnnData instance. The function
        creates a local copy of that AnnData and operates on it.
    cell_type_key : str
        Column name in adata.obs that contains cell type labels.
    markers : dict[str, dict[str, list[str]]]
        Marker specification mapping cell type -> {"positive": [...], "negative": [...]}
        - markers[ct]['positive'] should list genes expected to be present in source cells (ct2).
        - markers[ct]['negative'] should list genes expected to be absent in receiver cells (ct1).
    tables_key : str, optional
        Key to select the AnnData table from sdata.tables (default: "table").
    min_n_cells : int, optional
        Minimum number of cells required in each group (bordering or non_bordering)
        for a (ct1, ct2) pair to be tested (default: 20).
    min_n_transcripts : int, optional
        Minimum total transcript counts (sum across both groups) required for a gene
        to be kept in the results (default: 20).
    seed : int, optional
        Random seed for reproducible subsampling (default: 0).
    cell_centroid_x_key, cell_centroid_y_key : str, optional
        Column names in adata.obs that contain the X and Y cell centroids used to
        construct the spatial graph (default: "cell_centroid_x", "cell_centroid_y").
    inplace : bool, optional
        If True, store differential abundance results in `sdata.tables[tables_key].uns['diff_abundance']`.

    Returns
    -------
    de_results : pandas.DataFrame
        Concatenated differential abundance results for all tested (ct1, ct2) pairs.
        Columns include:
        - gene: gene name (string)
        - log2FC: reported log fold-change from rank_genes_groups
        - pval: p-value from the differential test
        - ct1: receiver cell type (string)
        - ct2: source cell type (string)
        - group1_size: number of ct1 cells bordering ct2 (int)
        - group2_size: number of ct1 cells not bordering ct2 (int)
        - transcript_counts_group1: total counts of the gene across group1 (int)
        - transcript_counts_group2: total counts of the gene across group2 (int)
        - transcript_counts_in_both_groups: sum of the two previous columns (int)
    summary : pandas.DataFrame
        A matrix (DataFrame) where rows are receiver cell types (ct1) and columns are
        source cell types (ct2). Each cell contains the number of genes passing the
        significance criteria for that ordered pair. Note: significance filtering uses
        the thresholds lfc_thresh and pval_thresh from the calling/global scope
        (see Notes).
    Raises
    ------
    ValueError
        If no differential expression results are produced (e.g., because no
        (ct1, ct2) pairs passed the min_n_cells / marker / transcript filters).
    """
    adata = sdata.tables[tables_key].copy()
    adata.obsm["spatial"] = adata.obs[[cell_centroid_x_key, cell_centroid_y_key]].to_numpy()

    # Replace NA cell types
    col = adata.obs[cell_type_key]
    if pd.api.types.is_categorical_dtype(col):
        if "Unknown" not in col.cat.categories:
            col = col.cat.add_categories(["Unknown"])
        adata.obs[cell_type_key] = col.fillna("Unknown")
    else:
        adata.obs[cell_type_key] = col.fillna("Unknown")

    # 1. Build spatial graph (Delaunay triangulation)
    sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
    G = adata.obsp["spatial_connectivities"].tocsr()

    types = np.asarray(adata.obs[cell_type_key])
    cell_types = np.unique(types)

    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    de_records = []

    # 2. Iterate over cell-type pairs
    for ct1 in tqdm(cell_types):
        idx_ct1 = np.where(types == ct1)[0]
        for ct2 in cell_types:
            if ct1 == ct2:
                continue

            # Find ct1 cells with ct2 neighbors
            neigh_counts = np.array([np.any(types[G.indices[G.indptr[i] : G.indptr[i + 1]]] == ct2) for i in idx_ct1])
            group1 = idx_ct1[neigh_counts]  # bordering
            group2 = idx_ct1[~neigh_counts]  # non-bordering
            group1_size, group2_size = len(group1), len(group2)

            if group1_size < min_n_cells or group2_size < min_n_cells:
                continue

            # Create condition labels
            adata.obs["_temp_condition"] = "not_used"
            adata.obs.iloc[group1, adata.obs.columns.get_loc("_temp_condition")] = "bordering"
            adata.obs.iloc[group2, adata.obs.columns.get_loc("_temp_condition")] = "non_bordering"

            # Differential test
            sc.tl.rank_genes_groups(
                adata,
                use_raw=False,
                groupby="_temp_condition",
                groups=["bordering"],
                reference="non_bordering",
                method="wilcoxon",
                pts=True,
            )

            res_dict = adata.uns["rank_genes_groups"]

            # ensure everything is in the right order
            genes = res_dict["names"]["bordering"]
            log2fc = res_dict["logfoldchanges"]["bordering"]
            pval = res_dict["pvals"]["bordering"]

            # transcript counts per group (while ensuring this is in the correct order)
            gene_idx = [adata.var_names.get_loc(g) for g in genes]
            transcript_counts_group1 = np.array(adata.raw.X[group1, :][:, gene_idx].sum(axis=0)).ravel()
            transcript_counts_group2 = np.array(adata.raw.X[group2, :][:, gene_idx].sum(axis=0)).ravel()

            res = pd.DataFrame(
                {
                    "gene": genes,
                    "log2FC": log2fc,
                    "pval": pval,
                    "ct1": ct1,
                    "ct2": ct2,
                    "group1_size": group1_size,
                    "group2_size": group2_size,
                    "transcript_counts_group1": transcript_counts_group1,
                    "transcript_counts_group2": transcript_counts_group2,
                    "transcript_counts_in_both_groups": transcript_counts_group1 + transcript_counts_group2,
                }
            )

            # removing rows with transcript counts lower than a certain minimum
            res = res[res["transcript_counts_in_both_groups"] >= min_n_transcripts]

            # only keeping genes that are positive in the source (ct2) and negative in the receiver (ct1)
            try:
                ct1_markers = markers[ct1]["negative"]
            except KeyError:
                if ct1 != "Unknown":
                    print(f"Could not find markers for cell type {ct1}")
                ct1_markers = []
            res = res[~res["gene"].isin(ct1_markers)]

            try:
                ct2_markers = markers[ct2]["positive"]
            except KeyError:
                if ct2 != "Unknown":
                    print(f"Could not find markers for cell type {ct2}")
                ct2_markers = []
            res = res[res["gene"].isin(ct2_markers)]

            de_records.append(res)

    # Combine results
    if not de_records:
        raise ValueError("No DE results produced — check thresholds or data sparsity.")
    de_results = pd.concat(de_records, ignore_index=True)

    # 4. Summarize significant DE genes
    # we are only interested in positive log2FC here, since we want to find genes that originate from spillover events
    sig = de_results.query("log2FC >= @lfc_thresh and pval <= @pval_thresh")
    summary = sig.groupby(["ct1", "ct2"]).size().unstack(fill_value=0)

    if inplace:
        if "diff_abundance" not in sdata.tables[tables_key].uns:
            sdata.tables[tables_key].uns["diff_abundance"] = {}
        sdata.tables[tables_key].uns["diff_abundance"]["de_results"] = de_results
        sdata.tables[tables_key].uns["diff_abundance"]["summary"] = summary

    return de_results, summary
