# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module: clustering stability
#
# Cells are usually clustered into cell types by their expression. With a good segmentation, the
# clusters should be well separated and stable: clustering a random subset of the cells should give
# similar clusters. Segmentation errors mix the expression of neighbouring cells and make clusters
# less distinct. The clustering stability module (`cs`) measures this.
#
# <center>
#  <img src='../_static/img/docs/clustering_stability.png' width='90%' />
# </center>
#
# We compare two segmentations of the same crop of a Xenium human breast cancer sample: the default
# Xenium segmentation and Proseg.

# %%
import warnings

import pandas as pd
import scanpy as sc

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

st = {
    "Xenium": segtraq.SegTraQ(segtraq.datasets.xenium_breast()),
    "Proseg": segtraq.SegTraQ(
        segtraq.datasets.proseg_breast(),
        tables_cell_id_key="cell",
        tables_area_key=None,
        tables_centroid_x_key="centroid_x",
        tables_centroid_y_key="centroid_y",
        points_cell_id_key="assignment",
        points_gene_key="gene",
        points_background_id=None,
        shapes_cell_id_key="cell",
        nucleus_shapes_cell_id_key="cell_id",
    ),
}

# %% [markdown]
# All metrics normalise and log-transform the counts, compute a PCA and a neighbour graph, and cluster
# the cells with Leiden (resolution 0.2 by default, set with `resolution`). They store their results
# in `table.uns`.
#
# ## Cluster connectedness
#
# `connectedness` is the fraction of each cell's neighbours in the expression graph that are in the
# same cluster. Higher is better.

# %%
for method, s in st.items():
    print(method, round(s.cs.connectedness(), 3))

# %% [markdown]
# ## Silhouette score
#
# `silhouette_score` compares how close each cell is to its own cluster and to the nearest other
# cluster in expression space. It ranges from -1 to 1; higher is better.

# %%
for method, s in st.items():
    print(method, round(s.cs.silhouette_score(), 3))

# %% [markdown]
# ## Purity and adjusted Rand index
#
# These two metrics cluster five random subsets of 63 % of the cells and compare the clusterings:
#
# - `purity`: for each cluster of one clustering, the largest fraction of its cells that falls into
#   one cluster of another clustering. 1 means identical clusters.
# - `adjusted_rand_index`: the agreement of two clusterings, corrected for chance. 1 means identical
#   clusters, 0 means chance agreement.

# %%
for method, s in st.items():
    print(method, "purity:", round(s.cs.purity(), 3), "ARI:", round(s.cs.adjusted_rand_index(), 3))

# %% [markdown]
# ## Using cell types instead of clusters
#
# Connectedness and silhouette score can also use existing labels, for example cell types transferred
# from a reference, with `cell_type_key`. They then measure how well the cell types are separated.

# %%
adata_ref = segtraq.datasets.reference("breast")
for method, s in st.items():
    s.run_label_transfer(adata_ref, ref_cell_type="celltype_major")
    connectedness = s.cs.connectedness(cell_type_key="transferred_cell_type")
    silhouette = s.cs.silhouette_score(cell_type_key="transferred_cell_type")
    print(method, "connectedness:", round(connectedness, 3), "silhouette:", round(silhouette, 3))

# %% [markdown]
# A UMAP of each segmentation, coloured by the transferred cell types, shows how well the cell types
# separate. `segtraq.pl.umap` plots the UMAPs of several methods side by side; we compute the UMAPs
# with scanpy.

# %%
sc.settings.verbosity = 0
for s in st.values():
    adata = s.sdata.tables["table"]
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata.X = adata.layers["counts"]

_ = segtraq.pl.umap(st, color="transferred_cell_type", legend=True, cols=2)

# %% [markdown]
# ## Run the whole module
#
# `run_clustering_stability` computes all metrics in one call. With `adata_ref` and `ref_cell_type`
# (or an existing `cell_type_key`), connectedness and silhouette score are also computed for the
# cell types.

# %%
st_xenium = segtraq.SegTraQ(segtraq.datasets.xenium_breast())
st_xenium.run_clustering_stability(adata_ref=adata_ref, ref_cell_type="celltype_major")
pd.Series({k: v for k, v in st_xenium.sdata.tables["table"].uns.items() if isinstance(v, float)})
