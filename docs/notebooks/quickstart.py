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
# # Quickstart
#
# To follow along with this tutorial, you can download the data from [here](https://oc.embl.de/index.php/s/YSvZTt8AArh4c5a).

# # Installation
#
# You can install SegTraQ via pip:
#
# ```bash
# pip install segtraq
# ```

# %% [markdown]
# Let's start by importing the required packages and setting the number of CPU cores to use for parallel processing.

# %%
import warnings

import anndata as ad
import spatialdata as sd

import segtraq

# Suppress specific warnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)

# Use all available CPU cores
segtraq.settings.n_jobs = -1

# %% [markdown]
# To get started with SegTraQ, we first need to load our data into a
# `spatialdata` object.
# Spatialdata-io provides readers for commonly used technologies,
# such as 10x Genomics Xenium, Vizgen MERSCOPE, and Nanostring CosMx.
# Segmentation tools such as Proseg also provide the option to
# export their results in `spatialdata` format directly.
# For more information on this, please refer to the [io notebook](io.ipynb).


# %%
sdata = sd.read_zarr("../../data/xenium_5K_data/proseg3.zarr")
sdata

# %% [markdown]
# Next, we can create a `SegTraQ` object by passing the `spatialdata` object
# to the `SegTraQ()` constructor.
# This has the advantage that we only need to set keywords like cell IDs
# or transcript IDs once, and they will be used for all subsequent analyses.
# If you do not know how to set the keywords, you can just use
# `segtraq.SegTraQ(sdata)` and
# the constructor will tell you which keywords need to be set.
# More information on this is available in the [io notebook](io.ipynb).

# %%
st = segtraq.SegTraQ(
    sdata,
    images_key="image",  # where the image is stored
    points_cell_id_key="assignment",  # what cells are called in the transcript table
    points_background_id=None,  # what unassigned transcripts are called
    points_gene_key="gene",  # what genes are called in the transcript table
    points_qv_key=None,  # what transcript quality is called in the transcript table
    tables_area_key=None,  # what the cell areas are called in the cell table
    tables_cell_id_key="cell",  # what cells are called in the expression matrix
    shapes_cell_id_key="cell",  # what cells are called in the segmentation shapes
    tables_centroid_x_key="centroid_x",  # what the cell centroids are called
    tables_centroid_y_key="centroid_y",  # what the cell centroids are called
)

# %% [markdown]
# Now that we have a `SegTraQ` object, we can call `run_all()`
# to compute all SegTraQ metrics in one go.

# %%
st.run_all()

# %% [markdown]
# As you can see, SegTraQ raises a warning that some metrics could not be computed.
# This is because some metrics require additional information, such as an scRNA-seq
# reference dataset or a 3D segmentation.
# These can be passed into the `run_all()` method as keyword arguments.

# %%
# reference scRNA-seq dataset for computing supervised metrics
adata_ref = ad.read_h5ad("../../data/xenium_5K_data/BC_scRNAseq_Janesick.h5ad")

# additional arguments to compute volume metrics
volume_kwargs = {
    "run_ovrlpy": True,
    "heterotypic_overlap_kwargs": {
        "shapes_key_list": [
            "cell_boundaries_z0",
            "cell_boundaries_z1",
            "cell_boundaries_z2",
            "cell_boundaries_z3",
        ]
    },
}

# run_all() with additional arguments
st.run_all(adata_ref=adata_ref, ref_cell_type="celltype_major", volume_kwargs=volume_kwargs)

# %% [markdown]
# All metrics are written to the `spatialdata` object.

# %%
st.sdata.tables["table"]

# %% [markdown]
# You can now use these metrics to perform quality control and
# filtering of your data,
# for example by filtering out spurious samples, cells,
# or comparing segmentation methods.
# More details on the available metrics and how to use them can be found in
# the module-specific notebooks.

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
