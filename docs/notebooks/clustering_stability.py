# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: segtraq2_env
#     language: python
#     name: segtraq2_env
# ---

# %% [markdown]
# # Module: Clustering Stability
#
# When analyzing (spatial) transcriptomics data, cells are typically clustered
# into cell types based on their RNA expression. If the segmentation of a spatial transcriptomics
# dataset went well, we would assume that this clustering is somewhat stable,
# even if we only cluster on parts of the data. The `clustering stability` (`cs`)
# module includes some functions to apply clustering and assess its robustness.
# The figure below shows a bad clustering (lots of overlap) vs. a good clustering.
#
# <center>
#  <img src='../_static/img/docs/clustering_stability.png' width='90%' />
# </center>
#
# To follow along with this tutorial, you can download the data from
# [here](https://oc.embl.de/index.php/s/iGxVy8qtZnwHOju).

# %%
# %load_ext autoreload
# %autoreload 2

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# %%
import importlib
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import spatialdata as sd
import spatialdata_plot  # noqa: F401

import segtraq


# %%
def filter_zero_count_cells(adata: ad.AnnData) -> ad.AnnData:
    """
    Return a view of adata excluding cells with zero total counts.
    Does NOT modify the original object.
    """
    if sp.issparse(adata.X):
        total_counts = np.array(adata.X.sum(axis=1)).flatten()
    else:
        total_counts = adata.X.sum(axis=1)

    mask = total_counts > 0
    return adata[mask, :]


# %%
# example datasets on which we can compare performance
st_bidcell = segtraq.SegTraQ(
    sd.read_zarr("../../data/xenium_5K_data/bidcell.zarr"),
    images_key="image",
    points_background_id=0,
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
)

st_segger = segtraq.SegTraQ(
    sd.read_zarr("../../data/xenium_5K_data/segger.zarr"),
    images_key="image",
    tables_centroid_x_key="cell_centroid_x",
    tables_centroid_y_key="cell_centroid_y",
)

st_xenium = segtraq.SegTraQ(
    sd.read_zarr("../../data/xenium_5K_data/xenium.zarr"),
    images_key="image",
    tables_centroid_x_key="x_centroid",
    tables_centroid_y_key="x_centroid",
)

st_proseg2 = segtraq.SegTraQ(
    sd.read_zarr("../../data/xenium_5K_data/proseg2.zarr"),
    images_key="image",
    points_background_id=0,
    tables_area_key=None,
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_x",
)

st_dict = {"bidcell": st_bidcell, "segger": st_segger, "xenium": st_xenium, "proseg2": st_proseg2}

# %% [markdown]
# ## The problem
# Segmentation errors affect downstream analysis in spatial transcriptomics datasets.
# For example, let's consider the UMAPs of four different segmentations.
# You can see below that they differ substantially.

# %%
fig, axs = plt.subplots(1, 4, figsize=(8, 2))

# Flatten axs in case of single row/column
axs = axs.flatten()

for i, (method, st) in enumerate(st_dict.items()):
    # here, we do not want to alter the original data, so we create a deep copy
    sdata = st.sdata
    adata = sd.deepcopy(sdata.tables["table"])

    # this is all done internally by SegTraQ as well before computing the cs metrics,
    # so you are not required to run this manually
    # normalizing and log-transforming the counts
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    # computing a PCA and neighbors
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    adata = filter_zero_count_cells(adata)

    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    sc.pl.umap(
        adata,
        color="leiden",
        ax=axs[i],
        show=False,
        title=method,
        legend_loc=None,
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cluster Connectedness
#
# The cluster connectedness (CC) is a measure that looks at the compactness of clusters.
# To compute it, we use the neighborhood graph we computed with scanpy earlier.
# The connectedness then iterates over all cells and computes the number of neighbors with the
# same cluster assignment and divides it by the number of total neighbors.
#
# By default, we compute this metric for Leiden clustering at resolution 0.2.
# You can adjust this with the `resolution` parameter.

# %%
ccs = {}
for method, st in st_dict.items():
    ccs[method] = st.cs.cluster_connectedness(use_weights=True, leiden_kwargs={"flavor": "igraph"})
ccs

# %% [markdown]
# ## Silhouette Score
#
# A slightly more elaborate metric is the silhouette score.
# It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
# Values range from −1 to +1, where a high value indicates that the object is well matched to its
# own cluster and poorly matched to neighboring clusters.
#
# By default, we compute this metric for Leiden clustering at resolution 0.2.
# You can adjust this with the `resolution` parameter.

# %%
silhouette_scores = {}
for method, st in st_dict.items():
    silhouette_scores[method] = st.cs.silhouette_score(leiden_kwargs={"flavor": "igraph"})
silhouette_scores

# %% [markdown]
# ## Purity
#
# Another way to assess cluster stability is to cluster only on a subset of all cells.
# For example, if we randomly select 63% ($1 - e^{-1}$) of cells and then perform Leiden
# clustering on those, will our cells typically get assigned to the same cluster or to different ones?
#
# One way to assess this is by comparing the purity between two clusterings.
# For every cluster in clustering 1, we check how many other clusters it contains in clustering 2.
#
# A purity value of 1 means that the clusters are completely pure,
# whereas a value closer to 0 means that they are more mixed.
#
# We randomly select 63% of cells, perform clustering on them, and do this five times,
# to obtain five different clusterings. Then we compare them using the purity score.
#
# **Note:** this method will recompute the PCA based on a subset of features.
# If you will be using the PCA in the anndata later, you should recompute it on the whole data.

# %%
purities = {}
for method, st in st_dict.items():
    purities[method] = st.cs.purity(leiden_kwargs={"flavor": "igraph"})
purities

# %% [markdown]
# ## Adjusted Rand Index (ARI)
#
# The adjusted rand index (ARI) is another metric to determine the similarity of
# different clusterings (again, we create five clusterings based on random subsets of 63% of cells each).
# Just like the purity, its values range from 0 (no similarity, not a robust clustering) to 1
# (exactly the same clusters, high robustness).

# %%
aris = {}
for method, st in st_dict.items():
    aris[method] = st.cs.adjusted_rand_index(leiden_kwargs={"flavor": "igraph"})
aris

# %% [markdown]
# At the end, all of our metrics are stored in the `spatialdata` object.

# %%
st_dict["proseg2"].sdata.tables["table"]

# %% [markdown]
# ## Visualization
#
# Looking at numbers is one thing, but interpretation will be a lot easier if we visualize our results.
# The following couple of codeblocks demonstrate how the four methods compare.

# %%
# putting everything into a dataframe for easier plotting
results_df = pd.DataFrame(
    {
        "Method": list(ccs.keys()),
        "Connectedness": list(ccs.values()),
        "Silhouette Score": list(silhouette_scores.values()),
        "Purity": list(purities.values()),
        "ARI": list(aris.values()),
    }
)

# %%
# plotting the connectedness vs. silhouette score
plt.figure(figsize=(4, 3))
sns.scatterplot(data=results_df, x="Connectedness", y="Silhouette Score", hue="Method", style="Method", s=100)
plt.xlabel("Connectedness (↑)")
plt.ylabel("Silhouette Score (↑)")
plt.title("Connectedness vs. Silhouette Score")
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.legend(title="Method")
plt.show()

# %%
# plotting the purity vs. ARI
plt.figure(figsize=(4, 3))
sns.scatterplot(data=results_df, x="Purity", y="ARI", hue="Method", style="Method", s=100)
plt.xlabel("Purity (↑)")
plt.ylabel("ARI (↑)")
plt.title("Purity vs. ARI")
plt.legend(title="Method")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# ## Supervised cluster stability
#
# If you already have labels for your cells, e. g. from performing label transfer on your data,
# you might want to check how compact those clusters are.
# Here, we assess Cluster Connectedness and Silhouette score on clusters obtained via label transfer.
# We do this by simply specifying a `label_key` in the respective functions.

# %%
# Running label transfer from an scRNA-seq dataset
scRNAseq_data_path = Path("../../data/xenium_5K_data/BC_scRNAseq_Janesick.h5ad")
adata_ref = ad.read_h5ad(scRNAseq_data_path)
st = segtraq.SegTraQ(
    sd.read_zarr("../../data/xenium_5K_data/proseg2.zarr"),
    images_key=None,
    tables_area_key=None,
    points_background_id=0,
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
)
st.run_label_transfer(adata_ref, ref_cell_type="celltype_major", ref_raw_counts_layer="raw")
sc.pp.pca(st.sdata.tables["table"])
sc.pp.neighbors(st.sdata.tables["table"])

# %%
st.cs.cluster_connectedness(cell_type_key="transferred_cell_type")

# %%
st.cs.silhouette_score(cell_type_key="transferred_cell_type")

# %% [markdown]
# ## Session Info

# %%
print(importlib.metadata.version("spatialdata"))
print(importlib.metadata.version("spatialdata_plot"))
print(importlib.metadata.version("scanpy"))
