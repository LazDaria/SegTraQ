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


def test_distance_to_centroid_columns_and_nonnegative(sdata_new):
    gene = _pick_present_gene(sdata_new)

    out = st.ps.distance_to_centroid(
        sdata_new,
        genes=gene,
        centroid_region="cell",
        restrict_to_within_boundary=False,
        inplace=False,
    )

    assert f"distance_to_cell_centroid_norm_{gene}" in out.columns

    # distances are Euclidean => non-negative
    assert np.all(out[f"distance_to_cell_centroid_norm_{gene}"].to_numpy() >= 0)


def test_distance_to_centroid_restrict_within_boundary_smaller_or_equal(sdata_new):
    """
    When, restricting to within cell, distance to centroid should be equal or smaller.
    We check a weak, robust condition: median shouldn't increase substantially.
    """
    gene = _pick_present_gene(sdata_new)

    out_all = st.ps.distance_to_centroid(
        sdata_new,
        genes=gene,
        centroid_region="cell",
        restrict_to_within_boundary=False,
        inplace=False,
    )
    out_in = st.ps.distance_to_centroid(
        sdata_new,
        genes=gene,
        centroid_region="cell",
        restrict_to_within_boundary=True,
        inplace=False,
    )

    col = f"distance_to_cell_centroid_norm_{gene}"

    med_all = np.nanmedian(out_all[f"{col}"].to_numpy())
    med_in = np.nanmedian(out_in[f"{col}"].to_numpy())

    assert med_in <= med_all


def test_distance_to_centroid_invalid_region_raises(sdata_new):
    with pytest.raises(ValueError):
        st.ps.distance_to_centroid(sdata_new, centroid_region="bad_region", inplace=False)
