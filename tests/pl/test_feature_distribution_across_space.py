import matplotlib.pyplot as plt
import pytest

import segtraq


def test_feature_distribution_across_space(sdata_new):
    plot = segtraq.pl.feature_distribution_across_space(sdata_new, features=["transcript_counts"])
    # plot should be a list of lists of axes, one per spatial axis
    assert isinstance(plot, list), f"Expected plot to be a list, got {type(plot)}"
    assert all(isinstance(ax, plt.Axes) for sublist in plot for ax in sublist), (
        f"Expected all elements of plot to be matplotlib Axes, got {[type(ax) for sublist in plot for ax in sublist]}"
    )


def test_feature_distribution_across_space_invalid_axis(sdata_new):
    with pytest.raises(ValueError, match="Axis column"):
        segtraq.pl.feature_distribution_across_space(
            sdata_new, features=["transcript_counts"], axes=("invalid_axis", "y")
        )


def test_feature_distribution_across_space_invalid_feature(sdata_new):
    with pytest.raises(ValueError, match="not found in obs"):
        segtraq.pl.feature_distribution_across_space(sdata_new, features=["invalid_feature"])


def test_feature_distribution_across_space_non_numeric_feature(sdata_new):
    with pytest.raises(ValueError, match="not numeric and could not be converted"):
        segtraq.pl.feature_distribution_across_space(sdata_new, features=["segmentation_method"])


def test_feature_distribution_across_space_invalid_filter_size(sdata_new):
    with pytest.raises(AssertionError, match="Filter size must be positive."):
        segtraq.pl.feature_distribution_across_space(sdata_new, features=["transcript_counts"], filter_size=-5)
    with pytest.raises(AssertionError, match="Filter size should be odd for symmetric smoothing."):
        segtraq.pl.feature_distribution_across_space(sdata_new, features=["transcript_counts"], filter_size=20)
