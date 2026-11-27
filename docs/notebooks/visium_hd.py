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
# # Visium HD
#
# SegTraQ was built for imaging-based data, where every transcript has its own position. Visium HD
# measures counts in 2 µm bins instead. `segtraq.bins_to_transcripts` turns each count into one
# transcript at the centre of its bin and assigns it to the cell that contains the bin. After that,
# SegTraQ works as for imaging-based data.
#
# We use a 500 µm crop of a Visium HD human ovarian cancer sample (10x Genomics) with the cell
# segmentation of Space Ranger. The conversion with `bins_to_transcripts` is shown in the
# [data loading tutorial](io.ipynb); here we start from the converted object.
#
# This analysis is described in the SegTraQ manuscript (Fig. 5).

# %%
import warnings

import matplotlib.pyplot as plt

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores

# %%
sdata = segtraq.datasets.visium_hd_ovarian()
sdata

# %% [markdown]
# ## Create the SegTraQ object
#
# Element names in Visium HD objects start with the dataset name. There are no quality values and no
# z positions. We remove the probes that 10x marks as deprecated (gene names starting with
# `DEPRECATED_`); Space Ranger also leaves them out of its count matrices.

# %%
DATASET = "Visium_HD_Human_Ovarian_Cancer_FF"

st = segtraq.SegTraQ(
    sdata,
    images_key=f"{DATASET}_hires_image",
    tables_key="cell_segmentations",
    tables_area_key=None,
    tables_centroid_x_key=None,
    tables_centroid_y_key=None,
    points_z_key=None,
    shapes_key=f"{DATASET}_cell_segmentations",
    nucleus_shapes_key=f"{DATASET}_nucleus_segmentations",
    filter_kwargs={"min_qv": None, "control_prefixes": ["DEPRECATED_"]},
)

# %% [markdown]
# ## Baseline metrics
#
# The baseline module counts cells, transcripts and genes and describes the cell shapes.

# %%
st.run_baseline()

table = st.sdata.tables["cell_segmentations"]
for k in ["num_cells", "num_transcripts", "num_genes", "perc_unassigned_transcripts"]:
    value = float(table.uns[k])
    print(f"{k}: {value:,.0f}" if value.is_integer() else f"{k}: {value:.3g}")

# %% [markdown]
# Per-cell values are stored in `obs`. Areas are in the units of the cell shapes, here full-resolution
# image pixels; we convert them to µm² with the pixel size from Space Ranger (`scalefactors_json.json`).

# %%
MICRONS_PER_PIXEL = 0.2738
obs = table.obs.assign(cell_area_um2=table.obs["cell_area"] * MICRONS_PER_PIXEL**2)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (column, label) in zip(
    axes,
    [
        ("transcript_count", "Transcripts per cell"),
        ("gene_count", "Genes per cell"),
        ("cell_area_um2", "Cell area (µm²)"),
    ],
    strict=True,
):
    ax.hist(obs[column].dropna(), bins=50, color="#99a3b5")
    ax.set(xlabel=label, ylabel="Cells")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# Cells contain a few thousand transcripts, many more than in imaging-based data, because Visium HD
# measures the whole transcriptome. Other modules work the same way as for imaging-based data,
# for example `st.run_supervised(...)` with `segtraq.datasets.reference("ovarian")`; see the module
# tutorials.
