# this simply tests if run_region_similarity works without errors
# it also ensures that the results are stored in the segtraq object's tables
def test_run_region_similarity(segtraq_obj):
    segtraq_obj.run_region_similarity(inplace=True)

    # check that the results are stored in the tables
    obs = segtraq_obj.sdata.tables["table"].obs
    assert "iou" in obs.columns
    assert "similarity_nucleus_cell" in obs.columns
    assert "similarity_nucleus_cytoplasm" in obs.columns
    assert "similarity_center_border" in obs.columns
    assert "similarity_border_neighborhood" in obs.columns
    assert "border_admixture_score" in obs.columns
