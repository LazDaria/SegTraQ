# import pytest

# import segtraq as st


# def test_purity(sdata_new):
#     purity = st.cs.purity(sdata_new, resolution=1.0, key_prefix="leiden_subset")
#     assert isinstance(purity, float), "Purity should be a float"
#     assert 0 <= purity <= 1, "Purity should be in the range [0, 1]"
#     assert "mean_purity" in sdata_new.tables["table"].uns.keys(), (
#         "Mean purity should be stored in sdata_new.tables['table'].uns"
#     )


# def test_purity_invalid_frac_cells_subset(sdata_new):
#     with pytest.raises(ValueError):
#         st.cs.purity(
#             sdata_new,
#             resolution=1.0,
#             key_prefix="leiden_subset",
#             frac_cells_subset=1.5,  # Invalid fraction > 1
#         )
