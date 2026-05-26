import copy as cp

import pandas as pd
import scanpy as sc

import segtraq


def test_umap(segtraq_obj):
    segtraq_obj_tmp = cp.deepcopy(segtraq_obj)
    sdata_tmp = segtraq_obj_tmp.sdata
    adata = sdata_tmp.tables["table"]
    # normalizing and log-transforming the counts
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # computing a PCA and neighbors
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    # computing UMAP
    sc.tl.umap(adata)
    st_dict = {"test1": segtraq_obj_tmp, "test2": segtraq_obj_tmp}
    umap_results = segtraq.pl.umap(st_dict, color="transferred_cell_type")

    # check that the output is a DataFrame with the expected columns
    assert isinstance(umap_results, pd.DataFrame)
    assert set(umap_results.columns) == {"x", "y", "Segmentation Method", "transferred_cell_type"}
