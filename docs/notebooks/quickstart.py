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
# # Quickstart
#
# Install SegTraQ with pip:
#
# ```bash
# pip install segtraq
# ```
#
# For the RCTD contamination metric, install `pip install "segtraq[rctd]"`.
#
# This tutorial computes all SegTraQ metrics for a small example. The data are downloaded
# automatically the first time (see `segtraq.datasets`).

# %%
import warnings

import segtraq

warnings.filterwarnings("ignore")
segtraq.settings.n_jobs = 8  # number of CPU cores; -1 uses all

# %% [markdown]
# ## Load the data
#
# SegTraQ works on [SpatialData](https://spatialdata.scverse.org) objects with transcripts, cell
# boundaries and a cell x gene table. Here we load a 450 µm crop of a Xenium human breast cancer
# sample. To load your own data, see [Loading data](io.ipynb).

# %%
sdata = segtraq.datasets.xenium_breast()
sdata

# %% [markdown]
# ## Create the SegTraQ object
#
# The SegTraQ object holds the data and the names of its elements and columns, so that they are set
# only once. The defaults match Xenium data; for other data, pass the names as arguments. If a name
# is wrong, SegTraQ says which one.

# %%
st = segtraq.SegTraQ(sdata)

# %% [markdown]
# ## Compute all metrics
#
# `run_all` runs all modules. The supervised metrics need a single-cell RNA-seq reference with cell
# types; here we use a breast cancer reference.

# %%
adata_ref = segtraq.datasets.reference("breast")
st.run_all(adata_ref=adata_ref, ref_cell_type="celltype_major")

# %% [markdown]
# Modules that cannot run with the available data are skipped with a message. Slow metrics are off by
# default: RCTD contamination (`supervised_kwargs={"run_rctd": True}`) and the vertical signal
# integrity of ovrlpy (`volume_kwargs={"run_ovrlpy": True}`).
#
# ## Results
#
# Per-cell metrics are stored in `obs` of the table, sample-level metrics in `uns` and per-gene metrics
# in `var`.

# %%
table = st.sdata.tables["table"]
table.obs.iloc[:, 9:].head()

# %%
for key in ["num_cells", "perc_unassigned_transcripts", "mean_ari_leiden", "mean_purity_leiden"]:
    value = float(table.uns[key])
    print(f"{key}: {value:,.0f}" if value.is_integer() else f"{key}: {value:.3g}")

# %% [markdown]
# Write the object to disk to keep the metrics: `st.sdata.write("xenium_segtraq.zarr")`.
#
# ## Next steps
#
# - [Comparing segmentation methods](segmentation_comparison.ipynb), [finding low-quality
#   samples](multi_sample.ipynb) and [Visium HD](visium_hd.ipynb) show complete analyses.
# - The module tutorials explain each metric.
