import numpy as np
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


def test_membrane_distance_skewness_columns_and_nan_masking(sdata_new):
    gene = _pick_present_gene(sdata_new)

    # force NaNs by demanding too many transcripts
    out = st.ps.membrane_distance_skewness(
        sdata_new,
        genes=gene,
        min_transcripts=10_000,
        inplace=False,
    )

    col = f"skew_dist_to_cell_membrane_{gene}"
    assert col in out.columns

    # With min_transcripts huge, everything should be NaN (or almost everything)
    assert out[col].isna().mean() > 0.9


def test_membrane_distance_skewness_reasonable_range_when_computable(sdata_new):
    gene = "ERBB2"

    out = st.ps.membrane_distance_skewness(
        sdata_new,
        genes=gene,
        min_transcripts=5,
        inplace=False,
    )
    col = f"skew_dist_to_cell_membrane_{gene}"
    vals = out[col].dropna().to_numpy()

    # If nothing passes, skip rather than fail (small bbox can be sparse)
    if vals.size == 0:
        pytest.skip("No cells passed min_transcripts in this subset.")

    # Skewness can be positive/negative; just check it's finite and in reasonable range
    assert np.isfinite(vals).all()
    assert np.nanmax(np.abs(vals)) < 50  # very permissive sanity bound
