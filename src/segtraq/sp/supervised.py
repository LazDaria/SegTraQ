import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import squidpy as sq
from anndata import AnnData
from scipy import sparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from ..utils import (
    _align_query_reference_genes,
    _get_count_matrix,
    _get_genes,
    _get_segtraq_markers,
    merge_into_obs,
    merge_into_uns,
)
from .utils import _binary_detection_matrix


def mutually_exclusive_coexpression_rate(
    sdata,
    adata_ref: AnnData,
    ref_cell_type: str,
    markers: dict[str, dict[str, list[str]]] | None = None,
    tables_key: str = "table",
    tables_gene_key: str | None = None,
    query_gene_key: str | None = None,
    tables_raw_counts_layer: str | None = None,
    ref_gene_key: str | None = None,
    ref_raw_counts_layer: str | None = None,
    min_pos_frac: float = 0.25,
    max_ref_coexpression_ratio: float = 0.75,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Assess unexpected co-expression of reference-defined mutually exclusive markers.

    Candidate gene pairs are derived from reciprocal positive/negative marker
    relationships between pairs of reference cell types. Positive markers must
    be detected in at least `min_pos_frac` of cells of their corresponding
    reference cell type. Pairs that are positive markers of the same cell type
    are excluded.

    Candidate pairs are retained as mutually exclusive if their observed
    co-expression in the reference is at most `max_ref_coexpression_ratio`
    times the co-expression expected under independence.

    For each retained pair, co-expression in the spatial data is summarized
    using a 2x2 detection table. A one-sided Fisher's exact test evaluates
    whether the genes are positively associated in the spatial data
    (odds ratio > 1). P-values are corrected across all tested pairs using
    the Benjamini-Hochberg procedure.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression data.
    adata_ref : AnnData
        Reference AnnData object containing annotated cells.
    ref_cell_type : str
        Column in `adata_ref.obs` containing reference cell-type labels.
    markers : dict or None, default=None
        Mapping of cell types to positive and negative markers:
        `{cell_type: {"positive": list[str], "negative": list[str]}}`.
        If None, markers are loaded from
        `sdata.tables[tables_key].uns["segtraq_markers"]`.
    tables_key : str, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing the canonical gene
        identifiers used by SegTraQ. If None, `var_names` are used.
    query_gene_key : str or None, default=None
        Alternative column in `sdata.tables[tables_key].var` containing gene
        identifiers matching `adata_ref.var[ref_gene_key]`. If None,
        `tables_gene_key` is used.
    tables_raw_counts_layer : str or None, default=None
        Layer containing raw spatial counts. If None, `.X` is used.
    ref_gene_key : str or None, default=None
        Column in `adata_ref.var` containing gene identifiers matching the
        query gene identifiers. If None, `adata_ref.var_names` are used.
    ref_raw_counts_layer : str or None, default=None
        Layer containing raw reference counts. If None, `.X` is used.
    min_pos_frac : float, default=0.25
        Minimum fraction of cells of the corresponding reference cell type
        in which a positive marker must be detected.
    max_ref_coexpression_ratio : float, default=0.75
        Maximum ratio of observed to expected co-expression in the reference
        for a candidate pair to be considered mutually exclusive.
    inplace : bool, default=True
        If True, store the result in
        `sdata.tables[tables_key].uns["mutually_exclusive_coexpression_rate"]`.

    Returns
    -------
    pd.DataFrame
        One row per reference-defined mutually exclusive marker pair with
        columns `gene1`, `gene2`, `odds_ratio`, `pvalue`, `pvalue_adj`,
        `a`, `b`, `c`, and `d`.

        `a` is the number of cells detecting both genes, `b` gene1 only,
        `c` gene2 only, and `d` neither. Odds ratios greater than 1 indicate
        positive association in the spatial data. `pvalue` is the one-sided
        Fisher exact p-value and `pvalue_adj` its Benjamini-Hochberg-adjusted
        value.
    """
    adata = sdata.tables[tables_key]
    columns = ["gene1", "gene2", "odds_ratio", "pvalue", "pvalue_adj", "a", "b", "c", "d"]

    # Stored marker indices refer to the original table gene axis.
    markers = _get_segtraq_markers(
        adata=adata,
        markers=markers,
        tables_gene_key=tables_gene_key,
    )

    # Convert query and reference to the same canonical gene namespace/order.
    adata, adata_ref = _align_query_reference_genes(
        adata_q=adata,
        adata_ref=adata_ref,
        tables_gene_key=tables_gene_key,
        query_gene_key=query_gene_key,
        ref_gene_key=ref_gene_key,
    )

    X = _get_count_matrix(
        adata,
        layer=tables_raw_counts_layer,
        layer_arg="tables_raw_counts_layer",
    )
    X_ref = _get_count_matrix(
        adata_ref,
        layer=ref_raw_counts_layer,
        layer_arg="ref_raw_counts_layer",
    )

    gene_index = adata.var_names
    shared_genes = set(gene_index)
    n_cells = X.shape[0]
    n_ref_cells = X_ref.shape[0]

    # Pairs that are positive markers of any same cell type are not mutually
    # exclusive, irrespective of the stricter MECR detection filter below.
    co_positive_pairs = set()

    for marker_set in markers.values():
        positive = sorted(set(marker_set.get("positive", [])) & shared_genes)

        co_positive_pairs.update((g1, g2) for i, g1 in enumerate(positive) for g2 in positive[i + 1 :])

    # Restrict all markers to genes shared between query and reference.
    marker_sets = {
        ct: {
            "positive": set(m.get("positive", [])) & shared_genes,
            "negative": set(m.get("negative", [])) & shared_genes,
        }
        for ct, m in markers.items()
    }

    # Require robust detection of positive markers in their reference cell type.
    for ct, marker_set in marker_sets.items():
        genes = sorted(marker_set["positive"])
        if not genes:
            continue

        cell_mask = np.asarray(adata_ref.obs[ref_cell_type] == ct)

        if not cell_mask.any():
            marker_set["positive"].clear()
            continue

        gene_idx = gene_index.get_indexer(genes)
        det = _binary_detection_matrix(X_ref[cell_mask], gene_idx)
        detection_fraction = np.asarray(det.mean(axis=0)).ravel()

        marker_set["positive"] = {
            gene for gene, frac in zip(genes, detection_fraction, strict=False) if frac >= min_pos_frac
        }

    # Build reciprocal positive/negative candidate pairs.
    candidate_pairs = set()
    celltypes = list(marker_sets)

    for i, ct_a in enumerate(celltypes):
        pos_a = marker_sets[ct_a]["positive"]
        neg_a = marker_sets[ct_a]["negative"]

        for ct_b in celltypes[i + 1 :]:
            pos_b = marker_sets[ct_b]["positive"]
            neg_b = marker_sets[ct_b]["negative"]

            candidate_pairs.update(
                tuple(sorted((g_a, g_b))) for g_a in pos_a & neg_b for g_b in pos_b & neg_a if g_a != g_b
            )

    candidate_pairs -= co_positive_pairs

    # Confirm mutual exclusivity in the reference.
    mutually_exclusive_pairs = set()

    if candidate_pairs:
        candidate_genes = sorted({gene for pair in candidate_pairs for gene in pair})
        gene_to_idx = {gene: i for i, gene in enumerate(candidate_genes)}

        idx = gene_index.get_indexer(candidate_genes)
        det_ref = _binary_detection_matrix(X_ref, idx)
        # gene x gene co-expression matrix
        coexpr_ref = det_ref.T @ det_ref

        if sparse.issparse(coexpr_ref):
            coexpr_ref = coexpr_ref.tocsr()

        ref_marginals = np.asarray(coexpr_ref.diagonal()).ravel()

        for g1, g2 in candidate_pairs:
            i1 = gene_to_idx[g1]
            i2 = gene_to_idx[g2]

            n1 = int(ref_marginals[i1])
            n2 = int(ref_marginals[i2])

            if n1 == 0 or n2 == 0:
                continue

            observed = int(coexpr_ref[i1, i2])
            expected = n1 * n2 / n_ref_cells

            if observed / expected <= max_ref_coexpression_ratio:
                mutually_exclusive_pairs.add((g1, g2))

    # Test retained pairs for positive association in the spatial data.
    rows = []

    if mutually_exclusive_pairs:
        genes = sorted({gene for pair in mutually_exclusive_pairs for gene in pair})
        gene_to_idx = {gene: i for i, gene in enumerate(genes)}

        idx = gene_index.get_indexer(genes)
        det = _binary_detection_matrix(X, idx)
        coexpr = det.T @ det

        if sparse.issparse(coexpr):
            coexpr = coexpr.tocsr()

        marginals = np.asarray(coexpr.diagonal()).ravel()

        for g1, g2 in sorted(mutually_exclusive_pairs):
            i1 = gene_to_idx[g1]
            i2 = gene_to_idx[g2]

            n1 = int(marginals[i1])
            n2 = int(marginals[i2])
            a = int(coexpr[i1, i2])
            b = n1 - a
            c = n2 - a
            d = n_cells - a - b - c

            odds_ratio, pvalue = fisher_exact(
                [[a, b], [c, d]],
                alternative="greater",
            )

            rows.append(
                {
                    "gene1": g1,
                    "gene2": g2,
                    "odds_ratio": float(odds_ratio),
                    "pvalue": float(pvalue),
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                }
            )

    df = pd.DataFrame(rows)

    if len(df):
        df["pvalue_adj"] = multipletests(
            df["pvalue"],
            method="fdr_bh",
        )[1]
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    if inplace:
        merge_into_uns(
            sdata=sdata,
            tables_key=tables_key,
            updates={"mutually_exclusive_coexpression_rate": df},
        )

    return df


def neighbor_contamination(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]] | None = None,
    tables_key: str = "table",
    tables_raw_counts_layer: str | None = None,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    tables_gene_key: str | None = None,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
):
    """
    Compute local negative-marker contamination per cell and per source-target
    cell-type pair.

    A gene is considered a locally relevant contamination marker for a target
    cell if it is:
    1. a negative marker of the target cell type, and
    2. a positive marker of at least one neighboring source cell type.

    If `require_neighbor_expression=True`, the gene must additionally be detected
    in at least one neighboring cell of the corresponding source cell type.

    Per-cell outputs (written to `adata.obs`):
        - contamination_counts:
            Total counts of locally relevant negative-marker genes in the focal cell.
        - contamination_strength:
            contamination_counts divided by the total transcript counts of the focal cell.
            This estimates the fraction of assigned transcripts that correspond to
            plausible local contamination.

    Source-target summaries (written to `adata.uns`):
        - contamination_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the fraction
            of evaluable target cells of type c_tgt that contain at least one locally
            relevant negative marker associated with source type c_src.
        - contamination_strength_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the mean
            source-specific contamination strength across evaluable target cells
            of type c_tgt with nonzero total counts.
        - contamination_evaluable_cells_matrix:
            Directed source-to-target matrix. Entry (c_src, c_tgt) is the number
            of target cells for which contamination from source type c_src could
            be evaluated.

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict or None, default=None
        Mapping of cell types to positive and negative markers:
        {cell_type: {"positive": list[str], "negative": list[str]}}.
        If None, markers are loaded from `adata.uns["segtraq_markers"]`.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_raw_counts_layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts.
        If a layer is specified, it must exist and contain count-like values.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is
        expressed in at least one neighboring cell of the source type.
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    per_cell_df : pandas.DataFrame
        Per-cell contamination counts and contamination strength.

    contamination_matrix_df : pandas.DataFrame
        Source-target matrix with the fraction of evaluable target cells contaminated
        by each source cell type.

    contamination_strength_matrix_df : pandas.DataFrame
        Source-target matrix with mean source-specific contamination strength.

    contamination_evaluable_cells_matrix_df : pandas.DataFrame
        Source-target matrix with the number of evaluable target cells per pair.
    """
    contamination_matrix_key = "contamination_matrix"
    contamination_strength_matrix_key = "contamination_strength_matrix"
    contamination_evaluable_cells_matrix_key = "contamination_evaluable_cells_matrix"

    # load expression matrix and metadata
    adata = sdata.tables[tables_key]

    markers = _get_segtraq_markers(
        adata=adata,
        markers=markers,
        tables_gene_key=tables_gene_key,
    )

    X = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = _get_genes(adata=adata, gene_key=tables_gene_key)
    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X_dense.shape[0]
    total_counts = np.asarray(X_dense.sum(axis=1)).ravel()

    # compute neighborhood graph if missing
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key={neighbors_key} missing; computing Delaunay neighbors.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors_delaunay(adata)

    # extract neighbor indices
    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # marker sets
    positive_sets = {ct: set(m.get("positive", [])) for ct, m in markers.items()}
    negative_sets = {ct: set(m.get("negative", [])) for ct, m in markers.items()}

    all_cts = sorted({ct for ct in cell_types if not pd.isna(ct)})

    # precompute relevant genes for each directed source-target pair:
    # genes that are negative in the target and positive in the source
    type_pair_genes: dict[tuple[str, str], np.ndarray] = {}

    for c_tgt in all_cts:
        neg = negative_sets.get(c_tgt, set())

        for c_src in all_cts:
            genes = list(neg & positive_sets.get(c_src, set()))
            if not genes:
                continue

            idx = var_index.get_indexer(genes)
            idx = idx[idx >= 0]

            if idx.size:
                type_pair_genes[(c_src, c_tgt)] = idx

    # per-cell outputs
    contamination_counts = np.full(n_cells, np.nan, dtype=float)
    contamination_strength = np.full(n_cells, np.nan, dtype=float)

    # source-target accumulators
    pair_hit_cells = defaultdict(int)
    pair_evaluable_cells = defaultdict(int)
    pair_strength_sum = defaultdict(float)

    # iterate over target cells.
    for i, c_tgt in enumerate(cell_types):
        if pd.isna(c_tgt) or c_tgt not in negative_sets:
            continue

        if total_counts[i] == 0:
            continue

        nbs = neighbor_indices[i]
        if len(nbs) == 0:
            continue

        x_i = X_dense[i, :]
        nb_cts = cell_types[nbs]

        used_genes_cell = set()
        pair_counts_this_cell = defaultdict(float)

        # evaluate contamination separately for each neighboring source type
        for c_src in {ct for ct in nb_cts if not pd.isna(ct)}:
            pair = (c_src, c_tgt)
            gene_idx = type_pair_genes.get(pair)

            if gene_idx is None:
                continue

            nb_src = nbs[nb_cts == c_src]
            X_nb_src = X_dense[np.ix_(nb_src, gene_idx)]

            valid_gene_idx = []
            for k, g_idx in enumerate(gene_idx):
                # optionally require expression in at least one neighboring
                # source cell
                if require_neighbor_expression and not (X_nb_src[:, k] > 0).any():
                    continue
                valid_gene_idx.append(g_idx)

            if not valid_gene_idx:
                continue

            # at least one locally relevant marker exists, so the cell is evaluable
            if np.isnan(contamination_counts[i]):
                contamination_counts[i] = 0.0

            for g_idx in valid_gene_idx:
                x_i_g = x_i[g_idx]

                # source-specific counts are accumulated per source-target pair
                pair_counts_this_cell[pair] += x_i_g

                # per-cell counts should count each gene only once, even if the
                # gene is relevant for multiple neighboring source types
                if g_idx not in used_genes_cell:
                    contamination_counts[i] += x_i_g
                    used_genes_cell.add(g_idx)

        # per-cell contamination strength
        if not np.isnan(contamination_counts[i]):
            contamination_strength[i] = contamination_counts[i] / total_counts[i]

        # source-target summaries for this target cell
        for pair, counts in pair_counts_this_cell.items():
            pair_evaluable_cells[pair] += 1

            if counts > 0:
                pair_hit_cells[pair] += 1

            source_strength = counts / total_counts[i]
            pair_strength_sum[pair] += source_strength

    per_cell_df = pd.DataFrame(
        {
            tables_cell_id_key: adata.obs[tables_cell_id_key],
            "contamination_counts": contamination_counts,
            "contamination_strength": contamination_strength,
        }
    )

    # build source-target matrices
    idxmap = {ct: i for i, ct in enumerate(all_cts)}

    contamination_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)
    strength_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)
    evaluable_mat = np.full((len(all_cts), len(all_cts)), np.nan, dtype=float)

    for (c_src, c_tgt), n_eval in pair_evaluable_cells.items():
        row = idxmap[c_src]
        col = idxmap[c_tgt]

        evaluable_mat[row, col] = n_eval
        contamination_mat[row, col] = pair_hit_cells[(c_src, c_tgt)] / n_eval

    for (c_src, c_tgt), total_strength in pair_strength_sum.items():
        n = pair_evaluable_cells[(c_src, c_tgt)]
        strength_mat[idxmap[c_src], idxmap[c_tgt]] = total_strength / n

    contamination_matrix_df = pd.DataFrame(
        contamination_mat,
        index=all_cts,
        columns=all_cts,
    )

    contamination_strength_matrix_df = pd.DataFrame(
        strength_mat,
        index=all_cts,
        columns=all_cts,
    )

    contamination_evaluable_cells_matrix_df = pd.DataFrame(
        evaluable_mat,
        index=all_cts,
        columns=all_cts,
    )

    if inplace:
        merge_into_obs(
            sdata=sdata,
            tables_key=tables_key,
            df_to_merge=per_cell_df,
            tables_cell_id_key=tables_cell_id_key,
            df_cell_id_key=tables_cell_id_key,
        )
        merge_into_uns(
            sdata,
            tables_key=tables_key,
            updates={
                contamination_matrix_key: contamination_matrix_df,
                contamination_strength_matrix_key: contamination_strength_matrix_df,
                contamination_evaluable_cells_matrix_key: contamination_evaluable_cells_matrix_df,
            },
        )

    return (
        per_cell_df,
        contamination_matrix_df,
        contamination_strength_matrix_df,
        contamination_evaluable_cells_matrix_df,
    )


def marker_purity(
    sdata,
    cell_type_key: str,
    markers: dict[str, dict[str, list[str]]] | None = None,
    tables_key: str = "table",
    tables_raw_counts_layer: str | None = None,
    tables_cell_id_key: str = "cell_id",
    tables_centroid_x_key: str = "x_centroid",
    tables_centroid_y_key: str = "y_centroid",
    tables_gene_key: str | None = None,
    require_neighbor_expression: bool = True,
    neighbors_key: str = "spatial_connectivities",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell marker purity using balanced accuracy.

    For each cell of type c:
        - positive markers of c are expected to be expressed
        - relevant negative markers are negative markers of c that are also
          positive markers of neighboring cell types

    The score combines:
        - positive_marker_recall: fraction of expected positive markers expressed
          (analogous to recall/sensitivity at the marker-level)
        - negative_marker_avoidance: fraction of relevant negative markers avoided
          (analogous to specificity at the marker-level)
        - marker_balanced_accuracy: mean of sensitivity and specificity

    Parameters
    ----------
    sdata : SpatialData-like
        Must contain `tables[tables_key]` as an AnnData with expression and `.obs` metadata.
    cell_type_key : str
        Column in the AnnData `.obs` with cell-type labels.
    markers : dict or None, default=None
        Mapping of cell types to positive and negative markers:
        {cell_type: {"positive": list[str], "negative": list[str]}}.
        If None, markers are loaded from `adata.uns["segtraq_markers"]`.
    tables_key : str, optional, default="table"
        Key of the AnnData table in `sdata.tables`.
    tables_raw_counts_layer : str | None, optional
        Layer containing count data. If `None`, `adata.X` is used if it looks
        like counts.
        If a layer is specified, it must exist and contain count-like values.
    tables_cell_id_key : str, optional, default="cell_id"
        Column in the AnnData `.obs` with unique cell IDs.
    tables_centroid_x_key : str or None, optional, default="x_centroid"
        Column in the cell table with the x-coordinate of the cell centroid.
    tables_centroid_y_key : str or None, optional, default="y_centroid"
        Column in the cell table with the y-coordinate of the cell centroid.
    tables_gene_key : str or None, default=None
        Column in `sdata.tables[tables_key].var` containing gene identifiers.
        If `None`, `sdata.tables[tables_key].var_names` are used.
    require_neighbor_expression : bool, optional, default=True
        If True, contamination is only counted when the relevant gene is
        expressed in at least one neighboring cell of the source type.
    neighbors_key : str, optional, default="spatial_connectivities"
        Key in `adata.obsp` containing a cell x cell adjacency / connectivity
        matrix that defines the spatial neighborhood.
    inplace : bool, optional, default=True
        If True, store marker purity results in `sdata.tables[tables_key].obs`.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - ``positive_marker_recall``
            - ``negative_marker_avoidance``
            - ``marker_balanced_accuracy``
            - ``n_evaluated_positive_markers``
            - ``n_evaluated_negative_markers``
    """
    adata = sdata.tables[tables_key]

    markers = _get_segtraq_markers(
        adata=adata,
        markers=markers,
        tables_gene_key=tables_gene_key,
    )

    X = _get_count_matrix(adata, layer=tables_raw_counts_layer)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_index = _get_genes(
        adata=adata,
        gene_key=tables_gene_key,
    )

    cell_types = np.asarray(adata.obs[cell_type_key])
    n_cells = X_dense.shape[0]

    # Compute spatial neighborhood graph if it is not already present.
    if neighbors_key not in adata.obsp:
        warnings.warn(
            f"neighbors_key='{neighbors_key}' not found in adata.obsp. "
            "A neighborhood graph based on Delaunay will be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        adata.obsm["spatial"] = adata.obs[[tables_centroid_x_key, tables_centroid_y_key]].to_numpy()
        sq.gr.spatial_neighbors_delaunay(adata)

    G = adata.obsp[neighbors_key]
    if sparse.issparse(G):
        G = G.tocsr()
        neighbor_indices = [G[i].indices for i in range(n_cells)]
    else:
        G = np.asarray(G)
        neighbor_indices = [np.where(G[i] > 0)[0] for i in range(n_cells)]

    # Keep only markers that are present in the spatial expression matrix.
    pos_sets = {ct: set(m.get("positive", [])) & set(var_index) for ct, m in markers.items()}
    neg_sets = {ct: set(m.get("negative", [])) & set(var_index) for ct, m in markers.items()}

    positive_recall = np.full(n_cells, np.nan, dtype=float)
    negative_avoidance = np.full(n_cells, np.nan, dtype=float)
    balanced_accuracy = np.full(n_cells, np.nan, dtype=float)
    n_pos_markers = np.zeros(n_cells, dtype=int)
    n_neg_markers = np.zeros(n_cells, dtype=int)

    for i, ct in enumerate(cell_types):
        if pd.isna(ct) or ct not in markers:
            continue

        pos_genes = pos_sets.get(ct, set())
        neg_all = neg_sets.get(ct, set())
        nbs = neighbor_indices[i]

        if not pos_genes:
            continue

        # Positive recall is computed globally for the focal cell type,
        # independent of the cell's neighborhood.
        pos_idx = var_index.get_indexer(list(pos_genes))
        pos_idx = pos_idx[pos_idx >= 0]

        n_pos_markers[i] = pos_idx.size
        pos_expr = X_dense[i, pos_idx] > 0
        positive_recall[i] = pos_expr.mean()

        # Negative avoidance is neighborhood-aware.
        if not neg_all or len(nbs) == 0:
            continue

        relevant_neg_genes = set()

        for nb_ct in pd.unique(cell_types[nbs]):
            if pd.isna(nb_ct) or nb_ct not in pos_sets:
                continue

            # Relevant negatives are focal negatives that are positive markers
            # of at least one neighboring cell type.
            candidate_genes = neg_all & pos_sets[nb_ct]
            if not candidate_genes:
                continue

            if not require_neighbor_expression:
                relevant_neg_genes.update(candidate_genes)
                continue

            nb_idx = nbs[cell_types[nbs] == nb_ct]

            # Optionally require the candidate gene to be expressed
            # in at least one neighbor of the corresponding source type.
            for g in candidate_genes:
                g_idx = var_index.get_loc(g)
                if (X_dense[nb_idx, g_idx] > 0).any():
                    relevant_neg_genes.add(g)

        if not relevant_neg_genes:
            continue

        neg_idx = var_index.get_indexer(list(relevant_neg_genes))
        neg_idx = neg_idx[neg_idx >= 0]

        n_neg_markers[i] = neg_idx.size
        neg_expr = X_dense[i, neg_idx] > 0

        negative_avoidance[i] = (~neg_expr).mean()
        balanced_accuracy[i] = 0.5 * (positive_recall[i] + negative_avoidance[i])

    result = pd.DataFrame(
        {
            tables_cell_id_key: adata.obs[tables_cell_id_key],
            "positive_marker_recall": positive_recall,
            "negative_marker_avoidance": negative_avoidance,
            "marker_balanced_accuracy": balanced_accuracy,
            "n_evaluated_positive_markers": n_pos_markers,
            "n_evaluated_negative_markers": n_neg_markers,
        },
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
