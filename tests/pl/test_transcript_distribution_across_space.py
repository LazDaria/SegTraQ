import matplotlib.pyplot as plt
import pytest

import segtraq as st


def test_transcript_distribution_across_space(sdata_new):
    plot = st.pl.transcript_distribution_across_space(sdata_new)
    # plot should be a list of axes, one per spatial axis
    assert isinstance(plot, list)
    assert all(isinstance(ax, plt.Axes) for ax in plot)


def test_transcript_distribution_across_space_invalid_axis(sdata_new):
    with pytest.raises(ValueError, match="Requested axis"):
        st.pl.transcript_distribution_across_space(sdata_new, axes=("invalid_axis", "y"))
