import numpy as np
import pandas as pd
import pytest

import segtraq as st


def _pick_present_gene(sdata, points_key="transcripts", points_gene_key="feature_name", n=2000):
    """
    Pick a gene that is actually present in the (possibly subsetted) points table.
    Works even if the points are backed by dask.
    """
    pts = sdata.points[points_key]
    # sample a small chunk
    sample = pts.head(n)
    if hasattr(sample, "compute"):
        sample = sample.compute()
    gene = sample[points_gene_key].dropna().astype(str).iloc[0]
    return gene


def _pick_cell_type(sdata, tables_key="table", cell_type_key="transferred_cell_type"):
    """
    Pick a cell type label present in the subset.
    """
    obs = sdata.tables[tables_key].obs
    if cell_type_key not in obs.columns:
        pytest.skip(f"{cell_type_key} not in sdata.tables['{tables_key}'].obs columns.")
    vals = obs[cell_type_key].dropna().astype(str)
    if vals.empty:
        pytest.skip(f"No non-null values in {cell_type_key}.")
    return vals.iloc[0]


def test_percentage_points_compartments_invariants(sdata_new):
    gene = _pick_present_gene(sdata_new)

    out = st.ps.percentage_points_compartments(
        sdata_new,
        genes=gene,
        predicate="intersects",
        inplace=False,
    )

    # Required columns exist (gene-specific naming)
    feature = gene
    expected = {
        f"n_total_{feature}",
        f"n_outside_cell_{feature}",
        f"n_in_nucleus_overlap_{feature}",
        f"n_in_cytoplasm_{feature}",
        f"pct_outside_cell_{feature}",
        f"pct_nucleus_{feature}",
        f"pct_cytoplasm_{feature}",
    }
    assert expected.issubset(set(out.columns))

    # Counts should be integers (after fillna + astype in your code)
    for c in [
        f"n_total_{feature}",
        f"n_outside_cell_{feature}",
        f"n_in_nucleus_overlap_{feature}",
        f"n_in_cytoplasm_{feature}",
    ]:
        assert pd.api.types.is_integer_dtype(out[c])

    # Invariants:
    # inside cell = total - outside
    inside = out[f"n_total_{feature}"] - out[f"n_outside_cell_{feature}"]
    # inside must equal nucleus_overlap + cytoplasm
    assert np.all(
        inside.to_numpy() == (out[f"n_in_nucleus_overlap_{feature}"] + out[f"n_in_cytoplasm_{feature}"]).to_numpy()
    )

    # Percentages sum to ~100 for cells where denominator > 0
    mask = out[f"n_total_{feature}"] > 0
    pct_sum = (
        out.loc[mask, f"pct_outside_cell_{feature}"]
        + out.loc[mask, f"pct_nucleus_{feature}"]
        + out.loc[mask, f"pct_cytoplasm_{feature}"]
    )
    assert np.allclose(pct_sum.to_numpy(), 100.0, rtol=0, atol=1e-6)


def test_percentage_points_compartments_gene_filter_reduces_totals(sdata_new):
    out_all = st.ps.percentage_points_compartments(sdata_new, genes=None, inplace=False)
    gene = _pick_present_gene(sdata_new)
    out_gene = st.ps.percentage_points_compartments(sdata_new, genes=gene, inplace=False)

    # total counts for a specific gene should be <= total counts for all genes (per cell)
    # (not every cell will have that gene; align by index)
    common_idx = out_all.index.intersection(out_gene.index)
    assert not common_idx.empty

    assert np.all(
        out_gene.loc[common_idx, f"n_total_{gene}"].to_numpy()
        <= out_all.loc[common_idx, "n_total_all_genes"].to_numpy()
    )


def test_percentage_points_compartments_cell_type_filter_runs(sdata_labeled):
    ct = _pick_cell_type(sdata_labeled)
    gene = _pick_present_gene(sdata_labeled)

    out = st.ps.percentage_points_compartments(
        sdata_labeled,
        genes=gene,
        cell_type_key="transferred_cell_type",
        cell_type_query=ct,
        inplace=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] >= 1
