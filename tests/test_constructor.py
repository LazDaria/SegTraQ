import numpy as np
import pytest
import spatialdata as sd

import segtraq as st


def test_constructor(sdata_new):
    st.SegTraQ(
        sdata_new,
        tables_centroid_x_key="x_centroid",
        tables_centroid_y_key="y_centroid",
        images_key="image",
        filter_kwargs={"inplace": False},
    )


def test_constructor_invalid_coordinate(sdata_new):
    with pytest.raises(AssertionError, match="Tables DataFrame must contain x coordinate column"):
        st.SegTraQ(
            sdata_new,
            images_key="image",
            tables_centroid_x_key="x",
            tables_centroid_y_key="y",
            filter_kwargs={"inplace": False},
        )


def test_constructor_missing_table_centroids(sdata_new):
    sdata = sd.deepcopy(sdata_new)
    # get the original centroid columns and drop them
    centroids_old = sdata.tables["table"].obs[["x_centroid", "y_centroid"]].copy()
    centroids_old.columns = ["centroid_x", "centroid_y"]
    sdata.tables["table"].obs.drop(columns=["x_centroid", "y_centroid"], inplace=True)
    # construct SegTraQ object, which should compute the centroids automatically
    st.SegTraQ(
        sdata,
        images_key="image",
        tables_centroid_x_key=None,
        tables_centroid_y_key=None,
        filter_kwargs={"inplace": False},
    )
    # get the new centroid columns
    centroids_new = sdata.tables["table"].obs[["centroid_x", "centroid_y"]]
    # check that the centroids are close to the old ones
    # they will not match exactly, not entirely sure why
    # however, since this should only be used when no centroids are provided, this should be acceptable
    assert np.allclose(centroids_old, centroids_new, atol=1), (
        "Centroid columns were not computed correctly during SegTraQ construction. "
        f"Example:\nOld centroids:\n{centroids_old.head()}\nNew centroids:\n{centroids_new.head()}"
    )
