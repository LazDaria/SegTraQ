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
# # Module: baseline
#
# The baseline module (`bl`) gives a first overview of a segmentation: the number of cells,
# transcripts and genes, how many transcripts are assigned to cells, and the size and shape of the
# cells.
#
# <center>
#  <img src='../_static/img/docs/baseline.png' width='90%' />
# </center>
#
# We use a crop of a Xenium human breast cancer sample. Each metric returns its values and also
# stores them in the table of the SpatialData object.

# %%
import warnings

import matplotlib.pyplot as plt

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

sdata = segtraq.datasets.xenium_breast()
st = segtraq.SegTraQ(sdata)

# %% [markdown]
# ## Sample-level metrics
#
# These metrics give one number per sample. They are stored in `table.uns`.

# %%
print("cells:", st.bl.num_cells())
print("transcripts:", st.bl.num_transcripts())
print("genes:", st.bl.num_genes())
print("unassigned transcripts (%):", round(st.bl.perc_unassigned_transcripts(), 2))

# %% [markdown]
# Transcripts and genes are counted after filtering: by default, SegTraQ removes control probes and
# transcripts with a quality value below 20.
#
# ## Cell-level metrics
#
# These metrics give one value per cell. They are stored in `table.obs`.

# %%
transcripts_per_cell = st.bl.transcripts_per_cell()
transcripts_per_cell.head()

# %%
genes_per_cell = st.bl.genes_per_cell()
transcript_density = st.bl.transcript_density()  # transcripts per µm² of cell area
mean_per_gene = st.bl.mean_transcripts_per_gene_per_cell()  # over the genes detected in a cell
genes_per_cell.head()

# %% [markdown]
# `morphological_features` describes the size and shape of each cell. Use `features_to_compute` to
# compute only some of them; `elongation` and `eccentricity` take longest.

# %%
morphology = st.bl.morphological_features()
morphology.head()

# %% [markdown]
# The distributions over all cells:

# %%
obs = st.sdata.tables["table"].obs
columns = {
    "transcript_count": "Transcripts per cell",
    "gene_count": "Genes per cell",
    "transcript_density": "Transcripts per µm²",
    "mean_transcripts_per_gene": "Transcripts per detected gene",
    "cell_area": "Cell area (µm²)",
    "circularity": "Circularity",
    "solidity": "Solidity",
    "elongation": "Elongation",
}
fig, axes = plt.subplots(2, 4, figsize=(13, 5))
for ax, (column, label) in zip(axes.flat, columns.items(), strict=True):
    values = obs[column].dropna()
    ax.hist(values, bins=50, color="#99a3b5")
    ax.axvline(values.median(), color="black", ls="--", lw=1)
    ax.set(xlabel=label, ylabel="Cells")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# Dashed lines show the medians.
#
# ## Gene-level metrics
#
# `perc_unassigned_transcripts_per_gene` gives the percentage of transcripts of each gene that are
# not assigned to a cell. It is stored in `table.var`. Genes that are often unassigned can point to
# cell types that the segmentation misses, or to transcripts outside cells.

# %%
per_gene = st.bl.perc_unassigned_transcripts_per_gene()
per_gene.sort_values("perc_unassigned", ascending=False).head()

# %% [markdown]
# In this crop, at most 2 % of the transcripts of any gene are unassigned.

# %% [markdown]
# ## Run the whole module
#
# `run_baseline` computes all of the above in one call.

# %%
st = segtraq.SegTraQ(segtraq.datasets.xenium_breast())
st.run_baseline()
st.sdata.tables["table"].obs.iloc[:, -12:].head()
