# def test_mecr_realdata_runs_and_stores_in_uns(sdata_3D_labeled, adata_ref):
#     # markers from your existing reference fixture
#     markers = st.markers_from_reference(adata_ref.copy(), cell_type_key="celltype_major")

#     df = st.sp.mutually_exclusive_coexpression_rate(
#         sdata=sdata_3D_labeled,
#         markers=markers,
#         tables_key="table",
#         inplace=True,
#     )

#     assert isinstance(df, pd.DataFrame)
#     print(df)
#     assert set(["gene1", "gene2", "odds_ratio", "pvalue", "a", "b", "c", "d"]).issubset(df.columns)

#     # Should store
#     tbl = sdata_3D_labeled.tables["table"]
#     assert "mutually_exclusive_coexpression_rate" in tbl.uns
#     assert tbl.uns["mutually_exclusive_coexpression_rate"].equals(df)
