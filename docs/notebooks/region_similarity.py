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
# # Module: region similarity
#
# If transcripts are spread evenly within a cell, different parts of the cell should have a similar
# gene expression. If they do not, transcripts of neighbouring cells may have been assigned to the
# cell. The region similarity module (`rs`) compares the transcripts of different parts of each cell.
#
# <center>
#  <img src='../_static/img/docs/region_similarity.png' width='90%' />
# </center>
#
# Small profiles look less similar by chance. SegTraQ therefore compares each observed similarity
# with the similarity expected from random permutations of the transcripts. Scores around 0 mean
# "as expected"; the p-values point to cells with an unusual score.
#
# We use a crop of a Xenium human breast cancer sample. Most metrics need nucleus boundaries.

# %%
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spatialdata_plot  # noqa: F401  (adds the .pl accessor to SpatialData)

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

sdata = segtraq.datasets.xenium_breast()
st = segtraq.SegTraQ(sdata)

# %% [markdown]
# We plot the results in a 150 µm window. The coordinate system is in DAPI pixels.

# %%
PX_PER_UM = 1 / 0.2125
x0, y0, size = 2330.125 * PX_PER_UM, 1755.0625 * PX_PER_UM, 150 * PX_PER_UM


def plot_cells(columns, titles, cmap="viridis", centred=False):
    window = st.sdata.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=[x0, y0],
        max_coordinate=[x0 + size, y0 + size],
        target_coordinate_system="global",
    )
    fig, axes = plt.subplots(1, len(columns), figsize=(4.5 * len(columns), 4))
    for ax, column, title in zip(axes, columns, titles, strict=True):
        norm = None
        if centred:  # symmetric colour scale around 0
            limit = np.nanpercentile(np.abs(window.tables["table"].obs[column].astype(float)), 99)
            norm = mpl.colors.Normalize(-limit, limit)
        window.pl.render_shapes(
            "cell_boundaries", color=column, cmap=cmap, norm=norm, outline_alpha=1, outline_width=0.3
        ).pl.show(ax=ax, title=title, colorbar_params={"label": ""})
        ax.set(xlim=(x0, x0 + size), ylim=(y0 + size, y0), xticks=[], yticks=[], xlabel="", ylabel="")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Overlap of cell and nucleus
#
# `match_nuclei_to_cells` matches each cell to the nucleus it overlaps most and measures the overlap:
#
# - `nucleus_fraction`: the fraction of the nucleus covered by the cell. A low value means that the
#   cell covers only part of a nucleus (oversegmentation).
# - `iou`: intersection over union of cell and nucleus. A low value means that the cell extends far
#   beyond the nucleus.
# - `num_nuclei`: the number of nuclei the cell overlaps.

# %%
st.rs.match_nuclei_to_cells()
st.sdata.tables["table"].obs[["num_nuclei", "iou", "nucleus_fraction"]].describe().round(2)

# %%
plot_cells(["iou", "nucleus_fraction"], ["IoU", "Nucleus fraction"])

# %% [markdown]
# Xenium grows cells from the nuclei, so the nucleus fraction is close to 1 for most cells.
#
# ## Nucleus and cell
#
# `similarity_nucleus_cell` compares the transcripts of the whole cell with those in its nucleus. A
# negative score means that the two are less similar than expected, for example because transcripts
# of a neighbouring cell were assigned to the cell.

# %%
_ = st.rs.similarity_nucleus_cell()

# %% [markdown]
# ## Nucleus and cytoplasm
#
# The whole cell contains the nucleus, so the nucleus-cell similarity is partly driven by the shared
# transcripts. `similarity_nucleus_cytoplasm` compares the nucleus with the rest of the cell
# directly. Low values can point to wrongly assigned transcripts, but also to genuine differences
# between nucleus and cytoplasm.

# %%
_ = st.rs.similarity_nucleus_cytoplasm()

# %% [markdown]
# ## Border admixture
#
# A cell border can differ from the cell centre for many reasons. `border_admixture_score` asks
# whether the border is better explained as a mixture of the cell centre and the cells around it:
#
# $$
# p_{\text{border}} \approx (1 - \alpha)\,p_{\text{centre}} + \alpha\,p_{\text{neighbourhood}}
# $$
#
# The score is the improvement of this mixture over the centre alone, minus the improvement expected
# by chance. Positive values mean that the border looks more like the neighbourhood than expected.
# This metric does not need nucleus boundaries.

# %%
scores = st.rs.border_admixture_score()
scores.dropna().head()

# %%
residuals = {
    "similarity_nucleus_cell": "Nucleus-cell similarity",
    "similarity_nucleus_cytoplasm": "Nucleus-cytoplasm similarity",
    "border_admixture_score": "Border admixture",
}
obs = st.sdata.tables["table"].obs
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (column, label) in zip(axes, residuals.items(), strict=True):
    ax.hist(obs[column].dropna(), bins=50, color="#99a3b5")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(xlabel=label, ylabel="Cells")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# In the window below, grey cells are not evaluated, for example because they have too few
# transcripts or no nucleus.

# %%
plot_cells(list(residuals), list(residuals.values()), cmap="coolwarm", centred=True)

# %% [markdown]
# ## Run the whole module
#
# `run_region_similarity` computes all of the above in one call.

# %%
st = segtraq.SegTraQ(segtraq.datasets.xenium_breast())
st.run_region_similarity()
st.sdata.tables["table"].obs.filter(regex="iou|nucleus_fraction|similarity|admixture").head()
