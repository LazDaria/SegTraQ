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
# # Module: point statistics
#
# Transcripts that spill over from a neighbouring cell lie near the cell border. The point
# statistics module (`ps`) describes where the transcripts of a cell lie relative to its centre, its
# membrane and its nucleus. This can be done for all transcripts or for selected genes and cell
# types.
#
# <center>
#  <img src='../_static/img/docs/point_statistics.png' width='90%' />
# </center>
#
# We compare two segmentations of the same crop of a Xenium human breast cancer sample: the default
# Xenium segmentation and Proseg. As an example gene we use ERBB2, a marker of DCIS2 tumour cells.

# %%
import warnings

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

# %% [markdown]
# The metrics can be restricted to cell types with `cell_type_key` and `cell_type_query`. We transfer
# cell types from a breast cancer reference.

# %%
adata_ref = segtraq.datasets.reference("breast")
for s in st.values():
    s.run_label_transfer(adata_ref, ref_cell_type="celltype_major")

# %% [markdown]
# ## Distance to the centroid
#
# `distance_to_centroid` measures how far the mean position of a cell's transcripts is from the
# centroid of the cell (or of its nucleus, `centroid_region="nucleus"`). The distance is divided by
# the square root of the cell area, so that cells of different size can be compared. Large values
# mean that the transcripts are shifted to one side of the cell.
#
# Proseg assigns transcripts that lie outside its cell boundaries. With
# `restrict_to_within_boundary=True`, only transcripts inside the cell are used.

# %%
for restrict in [False, True]:
    df = st["Proseg"].ps.distance_to_centroid(genes="ERBB2", restrict_to_within_boundary=restrict)
    print(f"restrict_to_within_boundary={restrict}:")
    print(df.describe().round(3).iloc[:, -1:], "\n")

# %% [markdown]
# ## Distance to the membrane
#
# `distance_to_membrane` gives the mean distance of a cell's transcripts to its boundary (or to its
# nucleus boundary, `membrane_region="nucleus"`). Distances are signed: positive inside the cell,
# negative outside.

# %%
df = st["Xenium"].ps.distance_to_membrane(genes="ERBB2", cell_type_query="DCIS2")
df.head()

# %% [markdown]
# ## Transcripts in compartments
#
# `percentage_transcripts_in_compartments` gives, per cell, the percentage of transcripts in the
# nucleus, in the cytoplasm and outside the cell. It needs nucleus boundaries.

# %%
for method, s in st.items():
    df = s.ps.percentage_transcripts_in_compartments(genes="ERBB2")
    print(method)
    print(df.filter(like="perc_").mean().round(1), "\n")

# %% [markdown]
# ## Membrane distance skewness
#
# `membrane_distance_skewness` measures the asymmetry of the distances of a cell's transcripts to its
# boundary, using transcripts inside the cell. Positive values mean that transcripts accumulate near
# the membrane, negative values that they accumulate in the interior. Cells with fewer than
# `min_transcripts` transcripts are not evaluated.

# %%
for method, s in st.items():
    df = s.ps.membrane_distance_skewness(min_transcripts=5)
    print(method, df.iloc[:, -1].describe().round(2).to_dict())

# %%
_ = segtraq.pl.boxplot(st, celltype_col="transferred_cell_type", value_key=df.columns[-1])

# %% [markdown]
# ## Run the whole module
#
# `run_point_statistics` computes all metrics in one call, for all genes and cell types by default.

# %%
st_xenium = segtraq.SegTraQ(segtraq.datasets.xenium_breast())
st_xenium.run_label_transfer(adata_ref, ref_cell_type="celltype_major")
st_xenium.run_point_statistics()
st_xenium.sdata.tables["table"].obs.iloc[:, -6:].head()
