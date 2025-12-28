import pytest

import segtraq as st


def test_constructor(sdata_new):
    st.SegTraQ(sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image")


def test_constructor_invalid_coordinate(sdata_new):
    with pytest.raises(AssertionError, match="Tables DataFrame must contain x coordinate column"):
        st.SegTraQ(sdata_new, images_key="image", tables_centroid_x_key="x", tables_centroid_y_key="y")
