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
# # Module: supervised
#
# In spatial data, transcripts can end up in the wrong cell, for example through diffusion or
# segmentation errors. Cells then partly look like their neighbours. The supervised module (`sp`)
# compares the cells with a single-cell RNA-seq reference with known cell types, and measures:
#
# - how well each cell matches the markers of its cell type,
# - how much it expresses markers of neighbouring cell types,
# - how mixed its expression profile is.
#
# <center>
#  <img src='../_static/img/docs/supervised.png' width='90%' />
# </center>
#
# We use a crop of a Xenium human breast cancer sample and a breast cancer scRNA-seq reference
# (Janesick et al.), subsampled to at most 500 cells per cell type.

# %%
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import spatialdata_plot  # noqa: F401  (adds the .pl accessor to SpatialData)

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

sdata = segtraq.datasets.xenium_breast()
st = segtraq.SegTraQ(sdata)
adata_ref = segtraq.datasets.reference("breast")  # raw counts in .X

# %% [markdown]
# ## Transfer cell types
#
# `run_label_transfer` gives each cell the reference cell type whose mean expression correlates best
# with the cell (Pearson correlation). Cells with fewer than 10 transcripts or 5 genes are not
# labelled; change this with `tx_min` and `gn_min`. The labels are stored in
# `obs["transferred_cell_type"]`.

# %%
st.run_label_transfer(adata_ref, ref_cell_type="celltype_major")
st.sdata.tables["table"].obs["transferred_cell_type"].value_counts()

# %% [markdown]
# The cell types in a 150 µm window:

# %%
PX_PER_UM = 1 / 0.2125  # the coordinate system is in DAPI pixels
x0, y0, size = 2330.125 * PX_PER_UM, 1755.0625 * PX_PER_UM, 150 * PX_PER_UM

cell_types = sorted(st.sdata.tables["table"].obs["transferred_cell_type"].dropna().unique())
window = st.sdata.query.bounding_box(
    axes=("x", "y"),
    min_coordinate=[x0, y0],
    max_coordinate=[x0 + size, y0 + size],
    target_coordinate_system="global",
)
colours = [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors[: len(cell_types)]] + ["#d9d9d9"]
palette = dict(zip([*cell_types, "unlabelled"], colours, strict=True))
table = window.tables["table"]
labels = table.obs["transferred_cell_type"].astype(object).fillna("unlabelled")
present = [c for c in palette if c in set(labels)]  # spatialdata-plot expects one colour per category present
table.obs["cell_type"] = pd.Categorical(labels, categories=present)
table.uns["cell_type_colors"] = [palette[c] for c in present]

fig, ax = plt.subplots(figsize=(6, 5))
window.pl.render_images("morphology_focus", cmap="gray").pl.render_shapes(
    "cell_boundaries", color="cell_type", fill_alpha=0.5, outline_alpha=1, outline_width=0.4
).pl.show(ax=ax, title="", legend_loc="right margin", colorbar=False)
ax.set(xlim=(x0, x0 + size), ylim=(y0 + size, y0), xticks=[], yticks=[], xlabel="", ylabel="")
plt.show()

# %% [markdown]
# ## Markers from the reference
#
# The metrics below use **positive markers** (genes expressed in a cell type) and **negative
# markers** (genes off in a cell type but on in others). `markers_from_reference` finds them by
# comparing all pairs of cell types in the reference (differential expression, `mode="de"`, or AUC,
# `mode="auc"`). A gene is a positive marker if it is up in at least half of the comparisons
# (`vote_fraction_pos`) and detected in at least 10 % of the cells of the type (`min_pos_frac`). It is
# a negative marker of a cell type if it is detected in at most 5 % of its cells (`max_neg_frac`)
# and up in another type. The markers are stored in `uns["segtraq_markers"]` and used by default.

# %%
markers = st.markers_from_reference(adata_ref, ref_cell_type="celltype_major")
pd.DataFrame({ct: {"positive": len(m["positive"]), "negative": len(m["negative"])} for ct, m in markers.items()}).T

# %% [markdown]
# Contamination from a cell type `a` into a cell type `b` is measured with genes that are positive
# markers of `a` and negative markers of `b`. The table counts these genes for each pair (rows:
# source `a`, columns: target `b`). With fewer than about 5 genes, the estimates are unstable; the
# thresholds above can then be relaxed.

# %%
overlap = pd.DataFrame(
    {b: {a: len(set(markers[a]["positive"]) & set(markers[b]["negative"])) for a in markers} for b in markers}
)
overlap

# %% [markdown]
# ## Marker balanced accuracy
#
# `marker_balanced_accuracy` is the mean of two fractions:
#
# - the fraction of positive markers of the cell's type that the cell expresses,
# - the fraction of relevant negative markers that the cell does not express. Relevant are negative
#   markers that are positive markers of the cell types around the cell.
#
# High values mean that the cell expresses its own markers and not those of its neighbours. The
# neighbours are found with a Delaunay graph of the cell centroids.

# %%
mba = st.sp.marker_balanced_accuracy(cell_type_key="transferred_cell_type")
mba.head()

# %% [markdown]
# ## Contamination strength
#
# `contamination_strength` is the fraction of a cell's transcripts that come from genes that are
# negative markers of its type and positive markers of a neighbouring type expressed nearby. It
# also returns, for each pair of cell types, the mean contamination of the target by the source
# (`uns["contamination_strength_matrix"]`) and the number of target cells that could be evaluated.

# %%
st.sp.contamination_strength(cell_type_key="transferred_cell_type")
matrix = st.sdata.tables["table"].uns["contamination_strength_matrix"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix.to_numpy(float), cmap="Reds")
ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90)
ax.set_yticks(range(matrix.shape[0]), matrix.index)
ax.set(xlabel="Target cell type", ylabel="Source cell type")
fig.colorbar(im, label="Mean contamination strength")
plt.show()

# %% [markdown]
# Contamination between two cell types depends on how often they are neighbours and on how many
# genes separate them, not only on the segmentation. Compare these values between methods on the
# same sample rather than between cell types.
#
# ## RCTD contamination
#
# `rctd_contamination` fits each cell as a mixture of all reference cell types with
# [RCTD](https://doi.org/10.1038/s41587-021-00830-w) (Python implementation
# [rctd-py](https://github.com/p-gueguen/rctd-py)). The score is 1 minus the weight of the main cell
# type: 0 means one cell type explains the cell, higher values mean a mixed profile. It needs neither
# transferred labels nor markers.
#
# RCTD is slow, especially without a GPU, so `run_supervised` and `run_all` skip it unless
# `run_rctd=True`.

# %%
rctd = st.sp.rctd_contamination(adata_ref=adata_ref, ref_cell_type="celltype_major")
rctd.head()

# %% [markdown]
# ## Mutually exclusive co-expression
#
# `mutually_exclusive_coexpression_rate` takes marker pairs that are rarely expressed together in
# the reference and tests, with Fisher's exact test, whether they are expressed together in the
# spatial cells more often than expected. An odds ratio above 1 with a small adjusted p-value means
# that the two genes, which should be exclusive, often appear in the same cell.

# %%
mecr = st.sp.mutually_exclusive_coexpression_rate(adata_ref=adata_ref, ref_cell_type="celltype_major")
mecr.sort_values("pvalue_adj").head()

# %% [markdown]
# ## Per cell type
#
# `segtraq.pl.boxplot` shows a per-cell metric for each cell type. It takes a dictionary of SegTraQ
# objects, so that several segmentation methods can be shown side by side.

# %%
for column in ["marker_balanced_accuracy", "contamination_strength", "rctd_contamination"]:
    _ = segtraq.pl.boxplot({"Xenium": st}, celltype_col="transferred_cell_type", value_key=column)

# %% [markdown]
# ## Run the whole module
#
# `run_supervised` runs label transfer, marker detection and all metrics in one call. Add
# `run_rctd=True` to include RCTD.

# %%
st = segtraq.SegTraQ(segtraq.datasets.xenium_breast())
st.run_supervised(adata_ref=adata_ref, ref_cell_type="celltype_major")
st.sdata.tables["table"].obs.filter(regex="cell_type|marker|contamination").head()
