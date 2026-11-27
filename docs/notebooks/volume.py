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
# # Module: volume
#
# Tissue sections are about 5-10 µm thick, so cells can lie on top of each other. A 2D segmentation
# then assigns the transcripts of two stacked cells to one cell. The volume module (`vl`) uses the z
# positions of the transcripts to detect such cells.
#
# <center>
#  <img src='../_static/img/docs/volume.png' width='90%' />
# </center>
#
# We compare two segmentations of the same crop of a Xenium human breast cancer sample: the default
# Xenium segmentation (2D) and Proseg, which models cells in 3D and stores one set of cell
# boundaries per z layer. Both objects contain the measured z positions.

# %%
import warnings

import matplotlib.pyplot as plt

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
adata_ref = segtraq.datasets.reference("breast")
for s in st.values():
    s.run_label_transfer(adata_ref, ref_cell_type="celltype_major")

# %% [markdown]
# ## Top-bottom similarity
#
# A single cell should have a similar expression at its top and at its bottom. Two stacked cells of
# different types do not. `similarity_top_bottom` takes the transcripts in the lowest and highest
# 30 % of each cell's z range and compares their expression (cosine similarity). As in the
# [region similarity module](region_similarity.ipynb), the score is the difference to the similarity
# expected by chance: negative values mean that top and bottom are less similar than expected, and
# the p-value points to cells with an unusually low similarity.
#
# The tissue is often not perfectly flat. By default, this global z drift is removed first
# (`correct_z_drift=True`).

# %%
for method, s in st.items():
    s.vl.similarity_top_bottom()
    obs = s.sdata.tables["table"].obs
    significant = (obs["similarity_top_bottom_p_value"] < 0.05).sum()
    print(f"{method}: {obs['similarity_top_bottom'].notna().sum()} cells evaluated, {significant} with p < 0.05")

# %%
fig, ax = plt.subplots(figsize=(5, 3))
for method, s in st.items():
    ax.hist(s.sdata.tables["table"].obs["similarity_top_bottom"].dropna(), bins=50, alpha=0.6, label=method)
ax.axvline(0, color="black", ls="--", lw=1)
ax.set(xlabel="Top-bottom similarity", ylabel="Cells")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# ## Heterotypic overlap
#
# Proseg stores cell boundaries per z layer (`cell_boundaries_z0`, `cell_boundaries_z1`, ...).
# `fraction_heterotypic_overlap` takes the largest boundary of each cell and measures how much of it
# overlaps with cells of another type in other z layers:
#
# - `heterotypic_overlap_area`: the overlapping area,
# - `heterotypic_overlap_fraction`: the same area divided by the cell area.
#
# This shows where Proseg places cells of different types on top of each other. It cannot be computed
# for 2D segmentations such as Xenium.

# %%
proseg = st["Proseg"]
layers = sorted(k for k in proseg.sdata.shapes if k.startswith("cell_boundaries_z"))
print(layers)
proseg.vl.fraction_heterotypic_overlap(shapes_key_list=layers)
proseg.sdata.tables["table"].obs[["heterotypic_overlap_area", "heterotypic_overlap_fraction"]].describe().round(2)

# %% [markdown]
# Very small cells can have a high fraction even if the overlap is small, so the area is often more
# informative.
#
# ## Vertical signal integrity
#
# [ovrlpy](https://doi.org/10.1101/2025.01.13.632601) compares the expression of a virtual top and
# bottom slice of the tissue at every position. The resulting vertical signal integrity (VSI) is
# high where top and bottom agree and low where cells overlap. It is computed from the transcripts
# alone, independent of the segmentation. `vertical_signal_integrity_per_cell` averages the VSI over
# the transcripts of each cell. We set the number of expression components to the number of cell
# types.

# %%
n_cell_types = st["Xenium"].sdata.tables["table"].obs["transferred_cell_type"].nunique()
for s in st.values():
    s.vl.vertical_signal_integrity_per_cell(ovrlpy_init_kwargs={"n_components": n_cell_types})

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, (method, s) in zip(axes, st.items(), strict=True):
    obs = s.sdata.tables["table"].obs
    ax.scatter(obs["vertical_signal_integrity"], obs["similarity_top_bottom"], s=3, alpha=0.5, color="#49566d")
    ax.set(title=method, xlabel="Mean VSI per cell")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Top-bottom similarity")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Run the whole module
#
# `run_volume` computes the top-bottom similarity and, if cell types and per-layer boundaries are
# available, the heterotypic overlap. ovrlpy is slow and runs only with `run_ovrlpy=True`.

# %%
st_proseg = segtraq.SegTraQ(
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
)
st_proseg.run_volume(
    adata_ref=adata_ref,
    ref_cell_type="celltype_major",
    heterotypic_overlap_kwargs={"shapes_key_list": layers},
)
st_proseg.sdata.tables["table"].obs.filter(regex="top_bottom|heterotypic").head()
