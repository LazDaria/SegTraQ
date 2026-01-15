import pytest

import segtraq as st


def test_filter_cells(sdata_new):
    st_obj = st.SegTraQ(
        sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image"
    )

    # all cells remain
    res = st_obj.filter_cells(
        col="x_centroid",
        func=lambda x: x > -1,  # all x_centroid are > -1
        inplace=False,
    )
    assert res.tables[st_obj.tables_key].shape[0] == st_obj.sdata.tables[st_obj.tables_key].shape[0], (
        f"All cells should remain after filtering. Before: {st_obj.sdata.tables[st_obj.tables_key].shape[0]}, "
        f"after: {res.tables[st_obj.tables_key].shape[0]}"
    )

    # some cells remain
    res = st_obj.filter_cells(
        col="x_centroid",
        func=lambda x: x > 800,  # some x_centroid are > 50
        inplace=False,
    )
    assert res.tables[st_obj.tables_key].shape[0] < st_obj.sdata.tables[st_obj.tables_key].shape[0], (
        f"Some cells should be filtered out. Before: {st_obj.sdata.tables[st_obj.tables_key].shape[0]}, "
        f"after: {res.tables[st_obj.tables_key].shape[0]}"
    )


def test_filter_cells_all_filtered(sdata_new):
    st_obj = st.SegTraQ(
        sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image"
    )

    with pytest.raises(AssertionError, match="Filtering removed all cells; no cells remain after filtering."):
        st_obj.filter_cells(
            col="x_centroid",
            func=lambda x: x > 2000,  # no x_centroid are > 2000
            inplace=False,
        )


def test_filter_cells_invalid_column(sdata_new):
    st_obj = st.SegTraQ(
        sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image"
    )
    with pytest.raises(AssertionError, match="Column 'invalid_column' not found in adata"):
        st_obj.filter_cells(
            col="invalid_column",
            func=lambda x: x > 50,
            inplace=False,
        )


def test_filter_cells_invalid_function(sdata_new):
    st_obj = st.SegTraQ(
        sdata_new, tables_centroid_x_key="x_centroid", tables_centroid_y_key="y_centroid", images_key="image"
    )
    with pytest.raises(TypeError, match="object is not callable"):
        st_obj.filter_cells(
            col="x_centroid",
            func="not_a_function",  # invalid function
            inplace=False,
        )
