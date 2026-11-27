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
# # Loading data: making methods comparable
#
# To compare segmentation methods, all objects should contain the same transcripts at the same
# positions, in the same coordinate system, with the same nuclei. Otherwise, differences in the
# metrics can come from the data preparation rather than from the segmentation.
#
# Segmentation methods write their results in different ways. This tutorial shows the differences
# between a Xenium and a Proseg object of the same sample and how to remove them. It then explains
# the choices made when converting Visium HD data. Read the [basic tutorial](io.ipynb) first.

# %%
import warnings

import numpy as np
import pandas as pd
from spatialdata.models import PointsModel
from spatialdata.transformations import get_transformation, set_transformation

import segtraq

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Xenium
#
# The Xenium object is the reference: its transcripts are the measured ones, and it contains the
# DAPI image and the nuclei that all methods start from.

# %%
xenium = segtraq.datasets.xenium_breast()
xenium

# %% [markdown]
# Its elements live in the coordinate system `global`, which is in pixels of the DAPI image. The
# image is stored in pixels; the translation places the cropped image at its position in the whole
# sample. Transcripts and boundaries are stored in µm, with a scale from µm to pixels (1 / 0.2125 µm
# per pixel):

# %%
for element in [xenium.images["morphology_focus"], xenium.points["transcripts"], xenium.shapes["cell_boundaries"]]:
    print(get_transformation(element))

# %% [markdown]
# (The GEO deposit of this sample lacks the files `cells.zarr.zip` and `experiment.xenium` that
# `spatialdata_io.xenium` needs. We therefore built the object with our own reader, using the same
# structure as spatialdata-io.)
#
# ## Proseg
#
# We load the object as written by Proseg (see the [basic tutorial](io.ipynb)) and compare it with
# the Xenium object.

# %%
proseg = segtraq.datasets.proseg_breast(raw=True)
transcripts = proseg.points["transcripts"].compute()
transcripts.head()

# %% [markdown]
# There are five differences.
#
# **1. Coordinate system.** Proseg stores everything in µm, and `global` is in µm too. We give the
# Proseg transcripts and boundaries the transformation of the Xenium object, so that they align with
# the DAPI image.

# %%
print(get_transformation(proseg.points["transcripts"]))
to_pixels = get_transformation(xenium.shapes["cell_boundaries"], get_all=True)

# %% [markdown]
# **2. Positions.** Proseg moves each transcript towards the cell it is assigned to. `x`, `y` and
# `z` are these moved positions; `observed_x`, `observed_y` and `observed_z` are the measured ones.
# Metrics such as the distance to the cell membrane should use the measured positions. We rename the
# columns, so that `x`, `y` and `z` are the measured positions.

# %%
transcripts = transcripts.rename(
    columns={
        "x": "repositioned_x",
        "y": "repositioned_y",
        "z": "repositioned_z",
        "observed_x": "x",
        "observed_y": "y",
        "observed_z": "z",
    }
)

# %% [markdown]
# **3. z positions.** Proseg rescales z to values between 0 and 1, while Xenium stores z in µm:

# %%
xenium_transcripts = xenium.points["transcripts"][["transcript_id", "x", "y", "z", "feature_name", "qv"]].compute()
pd.DataFrame({"Proseg": transcripts["z"].describe(), "Xenium": xenium_transcripts["z"].describe()}).round(2)

# %% [markdown]
# Both objects keep the Xenium `transcript_id`, so we take z from the Xenium object.
#
# A few transcripts have no match: their measured position is outside the crop, and Proseg moved
# them into it. They are not part of the crop, and we remove them.

# %%
transcripts["z"] = transcripts["transcript_id"].map(xenium_transcripts.set_index("transcript_id")["z"])
print(f"{transcripts['z'].isna().sum()} transcripts outside the crop")
transcripts = transcripts[transcripts["z"].notna()]

# %% [markdown]
# **4. Missing transcripts.** Proseg leaves out transcripts with a quality value below 20 and
# control probes. It also removes transcripts that are more than 60 µm away from the initial cell
# centroids (in this crop, almost none). In addition, a few hundred transcripts at the edge of the
# crop are missing because Proseg moved them out of the crop.

# %%
left_out = xenium_transcripts[~xenium_transcripts["transcript_id"].isin(transcripts["transcript_id"])]
print(f"Xenium: {len(xenium_transcripts)} transcripts, Proseg: {len(transcripts)}, left out: {len(left_out)}")
print(f"left out with a quality value below 20: {(left_out['qv'] < 20).sum()}")

# %% [markdown]
# SegTraQ removes low-quality transcripts and control probes by default too. We still add all
# left-out transcripts back, as transcripts outside cells, so that both objects contain the same
# transcripts. Then any filter, for example a different quality threshold, affects both methods in
# the same way.

# %%
left_out = left_out.rename(columns={"feature_name": "gene"}).assign(assignment=float("nan"), background=True)
transcripts = pd.concat([transcripts, left_out], ignore_index=True)
transcripts["gene"] = transcripts["gene"].astype("category")

proseg.points["transcripts"] = PointsModel.parse(
    transcripts,
    coordinates={"x": "x", "y": "y", "z": "z"},
    feature_key="gene",
    instance_key="assignment",
    transformations=to_pixels,
)
set_transformation(proseg.shapes["cell_boundaries"], to_pixels, set_all=True)
print(f"Proseg now has {len(transcripts)} transcripts, Xenium {len(xenium_transcripts)}")

# %% [markdown]
# **5. Image and nuclei.** Proseg starts from the Xenium nuclei, but does not store them. We add the
# DAPI image and the nucleus boundaries of the Xenium object. The nucleus boundaries keep the Xenium
# cell IDs, which link each nucleus to the Xenium cell it came from.

# %%
proseg.images["morphology_focus"] = xenium.images["morphology_focus"]
proseg.shapes["nucleus_boundaries"] = xenium.shapes["nucleus_boundaries"]
proseg

# %% [markdown]
# The Proseg object now contains the same transcripts, positions, image and nuclei as the Xenium
# object, so that SegTraQ can compute all metrics for both. When creating the SegTraQ object, pass
# `nucleus_shapes_cell_id_key="cell_id"`: the nuclei keep the Xenium cell IDs, which link each
# nucleus to the Xenium cell it came from (see the [segmentation comparison](segmentation_comparison.ipynb)).

# %% [markdown]
# The harmonised Proseg object of the whole crop can also be downloaded with
# `segtraq.datasets.proseg_breast()`. It was prepared in the same way, from the whole sample.
#
# ## Visium HD
#
# For Visium HD, `segtraq.bins_to_transcripts` creates the transcripts (see the
# [basic tutorial](io.ipynb)). Three points are worth knowing.
#
# **Which bins.** `spatialdata_io.visium_hd(..., filtered_counts_file=False)` reads all bins, not
# only those under the tissue. Other methods that start from the bins, such as Proseg, also use all
# bins, so this keeps the transcripts comparable.
#
# **Positions.** Each transcript is placed at the centre of its 2 µm bin. All transcripts of a bin
# share one position, and they are assigned to the cell whose boundary contains the bin centre.
# Distances below the bin size are not meaningful.
#
# **Genes.** The bin table lists all genes of the genome annotation, while the Space Ranger cell
# table contains only the targeted genes. Few of the extra genes have counts, and most of these
# counts come from probes that 10x marks as deprecated (names starting with `DEPRECATED_`):

# %%
visium = segtraq.datasets.visium_hd_ovarian(raw=True)
bins = visium.tables["square_002um"]
extra = bins.var_names.difference(visium.tables["cell_segmentations"].var_names)
counts = pd.Series(np.asarray(bins[:, extra].X.sum(axis=0)).ravel(), index=extra)
deprecated = extra.str.startswith("DEPRECATED_")
print(f"genes in the bins: {bins.n_vars}, in the cell table: {visium.tables['cell_segmentations'].n_vars}")
print(f"genes only in the bins: {len(extra)}, with counts: {(counts > 0).sum()}")
print(f"counts on these genes: {counts.sum():.0f} ({counts.sum() / bins.X.sum():.1%} of all counts)")
print(f"of these on deprecated probes: {counts[deprecated].sum():.0f}")

# %% [markdown]
# SegTraQ removes the deprecated probes with its control-probe filter (see the
# [Visium HD tutorial](visium_hd.ipynb)).
#
# ## Save the objects
#
# Write the harmonised objects to disk to reuse them in other analyses:
#
# ```python
# proseg.write("proseg_breast_crop.zarr")
# ```
