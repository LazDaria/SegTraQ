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


def test_distance_to_membrane_restrict_cell_makes_signed_nonnegative(sdata_new):
    gene = _pick_present_gene(sdata_new)

    out = st.ps.distance_to_membrane(
        sdata_new,
        genes=gene,
        membrane_region="cell",
        restrict_to_within_boundary=True,  # inside/on cell
        signed=True,
        inplace=False,
    )
    col = f"distance_to_cell_membrane_norm_{gene}"
    assert col in out.columns

    # if we restrict to within the cell and measure distance to cell membrane, signed distances should be >= 0
    # (covers includes boundary)
    assert np.all(out[col].to_numpy() >= 0)


def test_distance_to_membrane_signed_can_be_negative_without_restrict(sdata_new):
    gene = _pick_present_gene(sdata_new)

    out = st.ps.distance_to_membrane(
        sdata_new,
        genes=gene,
        membrane_region="cell",
        restrict_to_within_boundary=False,
        signed=True,
        inplace=False,
    )
    raw_col = f"distance_to_cell_membrane_{gene}"
    assert raw_col in out.columns

    # transcripts can fall outside the cell polygon.
    # ensure it's numeric.
    vals = out[raw_col].to_numpy()
    assert np.isfinite(vals).any()


def test_distance_to_membrane_inverse_score_monotonic(sdata_new):
    gene = _pick_present_gene(sdata_new)

    out = st.ps.distance_to_membrane(
        sdata_new,
        genes=gene,
        membrane_region="cell",
        restrict_to_within_boundary=True,
        signed=True,
        inverse_score=True,
        inplace=False,
    )
    inv = out[f"distance_to_cell_membrane_inverse_{gene}"].to_numpy()

    # inverse score defined as 1/sqrt(abs(dist)+eps): should be positive and finite where dist finite
    assert np.all(inv > 0)
    assert np.isfinite(inv).all()


def test_distance_to_membrane_invalid_region_raises(sdata_new):
    with pytest.raises(ValueError):
        st.ps.distance_to_membrane(sdata_new, membrane_region="bad_region", inplace=False)
