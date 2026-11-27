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
# # Loading data
#
# SegTraQ works on [SpatialData](https://spatialdata.scverse.org) objects. It needs:
#
# - **transcripts** (points) with their positions, genes and assigned cell IDs,
# - **cell boundaries** (shapes),
# - a **cell x gene table**.
#
# Some metrics also use **nucleus boundaries**, **z positions** of the transcripts or a **DAPI
# image**. Without them, these metrics are skipped:
#
# - without **nucleus boundaries**: the nucleus-dependent region similarity metrics (nucleus-cell
#   matching, nucleus-cell and nucleus-cytoplasm similarity) and the transcripts in compartments
#   (point statistics),
# - without **z positions**: the volume module,
# - without a **DAPI image**: nothing. The image is only needed to segment nuclei with
#   `segtraq.cellpose` when nucleus boundaries are missing.
#
# This tutorial loads three datasets in their default form: Xenium, Proseg and Visium HD. The
# [extended tutorial](io_extended.ipynb) explains how to make objects of different methods
# comparable.

# %%
import warnings

import segtraq

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Xenium
#
# A Xenium output folder is read with [spatialdata-io](https://spatialdata.scverse.org/projects/io):
#
# ```python
# import spatialdata_io
#
# sdata = spatialdata_io.xenium("path/to/xenium_output")
# ```
#
# Here we download a crop of a Xenium human breast cancer sample instead. (Its GEO deposit lacks
# some files that `spatialdata_io.xenium` needs, so we built the object with our own reader.)

# %%
xenium = segtraq.datasets.xenium_breast()
xenium

# %% [markdown]
# The object contains everything SegTraQ uses: transcripts with quality values and z positions, cell
# and nucleus boundaries, the cell x gene table and the DAPI image (`morphology_focus`). Its element
# and column names are the SegTraQ defaults.
#
# ## Proseg
#
# [Proseg](https://github.com/dcjones/proseg) writes a SpatialData object directly
# (`--output-spatialdata`). We load a crop of Proseg run on the same Xenium sample:

# %%
proseg = segtraq.datasets.proseg_breast(raw=True)
proseg

# %%
proseg.points["transcripts"].head()

# %% [markdown]
# Proseg uses its own column names: cell IDs are in `assignment` and genes in `gene`. Transcripts
# outside cells have a missing cell ID. The object has no image and no nucleus boundaries, so the
# nucleus-dependent metrics would be skipped. The [segmentation comparison](segmentation_comparison.ipynb)
# shows how to pass these names to SegTraQ.
#
# Note that Proseg moves transcripts towards the cells they are assigned to: `x`, `y` and `z` are
# the moved positions, `observed_x`, `observed_y` and `observed_z` the measured ones. Proseg also
# leaves out some transcripts. To compare Proseg with other methods, use the measured positions and
# all transcripts, as shown in the [extended tutorial](io_extended.ipynb).
#
# ## Visium HD
#
# Visium HD measures counts in 2 µm bins. Space Ranger also segments cells and assigns bins to them.
# spatialdata-io reads the bins and the cells:
#
# ```python
# import spatialdata_io
#
# sdata = spatialdata_io.visium_hd(
#     "path/to/spaceranger_output",
#     bin_size=2,
#     load_nucleus_segmentations=True,
#     filtered_counts_file=False,  # all bins, not only those under the tissue
# )
# ```
#
# We download a crop of a Visium HD human ovarian cancer sample, read this way:

# %%
visium = segtraq.datasets.visium_hd_ovarian(raw=True)
visium

# %% [markdown]
# There are no transcripts yet, only counts per bin (table `square_002um`).
# `segtraq.bins_to_transcripts` creates one transcript per count, placed at the centre of its bin. A
# transcript is assigned to the cell whose boundary contains the bin centre.

# %%
DATASET = "Visium_HD_Human_Ovarian_Cancer_FF"

visium = segtraq.bins_to_transcripts(
    visium,
    bins_tables_key="square_002um",
    cells_tables_key="cell_segmentations",
    bins_shapes_key=f"{DATASET}_square_002um",
    cell_shapes_key=f"{DATASET}_cell_segmentations",
    coordinate_system=DATASET,
    background_id="UNASSIGNED",
)
visium.points["transcripts"].head()

# %% [markdown]
# Visium HD has no z positions and no quality values, so the volume module would be skipped. The
# [Visium HD tutorial](visium_hd.ipynb) shows how to run SegTraQ on this object.
#
# ## Save the objects
#
# To keep a converted object, write it to disk and read it again later:
#
# ```python
# import spatialdata as sd
#
# visium.write("visium_hd_ovarian_crop.zarr")
# sdata = sd.read_zarr("visium_hd_ovarian_crop.zarr")
# ```
