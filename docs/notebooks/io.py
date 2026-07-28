# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: segtraq_py311 (3.11.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Preparation
#
# In order to use `SegTraQ`, you first need to get your data into a `SpatialData` object.
# While `spatialdata_io` has a lot of readers for different technologies,
# the constantly evolving landscape of segmentation methods makes it difficult to provide up-to-date,
# generalizable readers. To facilitate the process of getting your data into the correct format,
# we provide examples for some segmentation methods.
#
# To follow along with this tutorial, you can download the data from [here](https://oc.embl.de/index.php/s/iGxVy8qtZnwHOju).

# %%
import anndata as ad

ad_test = ad.read_h5ad("../../../data/xenium_5K_data/BC_scRNAseq_Janesick.h5ad")

# %%
import spatialdata_io  # noqa
import spatialdata_plot  # noqa
import segtraq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import anndata as ad
import tifffile as tiff
import numpy as np
import geopandas as gpd
import spatialdata as sd
import gzip
import shapely
import json

from pathlib import Path
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.sparse import csr_matrix
from spatialdata.models import PointsModel
from spatialdata.transformations import (
    get_transformation,
    set_transformation,
)


# %% [markdown]
# Below, we define a few custom functions that will be required for reading data below.


# %%
def labels_to_shapes(label_img: np.ndarray, simplify_tolerance: float | None = 0.5) -> gpd.GeoDataFrame:
    if label_img.ndim != 2:
        raise ValueError("Input label_img must be 2D.")

    lab = np.asarray(label_img)
    lab = lab.astype(np.int32, copy=False)
    mask = lab != 0

    geoms = []
    ids = []
    for geom_mapping, value in shapes(lab, mask=mask, connectivity=8):
        ids.append(int(value))
        geoms.append(shape(geom_mapping))

    gdf = gpd.GeoDataFrame({"cell_id": ids, "geometry": geoms}).set_index("cell_id")
    gdf = gdf.dissolve(by="cell_id", as_index=True)

    if simplify_tolerance and simplify_tolerance > 0 and not gdf.empty:
        gdf["geometry"] = shapely.simplify(gdf.geometry.values, simplify_tolerance, preserve_topology=True)

    return gdf


def crop(ws, bb_xmin, bb_ymin, bb_xmax, bb_ymax):
    return ws.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[bb_xmin, bb_ymin],
        max_coordinate=[bb_xmax, bb_ymax],
        target_coordinate_system="global",
    )


def read_geojson_gz(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return gpd.GeoDataFrame.from_features(data["features"])


# %% [markdown]
# Setting the data path:

# %%
data_path = Path("/g/huber/projects/CODEX/segtraq/data/20260113_Janesick_Replicate1/xenium")

# %% [markdown]
# ## Xenium
#
# For Xenium data, we can make use of the functions from `spatialdata_io`.
# We don't need `cells_as_circles`, `cell_labels`, `nucleus_labels` and `morphology_mip`
# for `SegTraQ`, so we will not read these data.

# %%
sdata_xenium = spatialdata_io.xenium(
    data_path, cells_as_circles=False, cells_labels=False, nucleus_labels=False, morphology_mip=False
)
sdata_xenium

# %% [markdown]
# For visualization, we crop the `SpatialData` object to zoom into a smaller region.

# %%
bb_xmin = 10000
bb_ymin = 12500
bb_w = 2500
bb_h = 2500
bb_xmax = bb_xmin + bb_w
bb_ymax = bb_ymin + bb_h

f, ax = plt.subplots(figsize=(5, 5))
sdata_xenium.pl.render_shapes("cell_boundaries").pl.show(ax=ax)
rect = patches.Rectangle((bb_xmin, bb_ymin), bb_w, bb_h, linewidth=5, edgecolor="red", facecolor="none")
ax.add_patch(rect)

# %% [markdown]
# Cropping currently suffers from performance issues, and can take tens of minutes on large datasets.
# This will be improved in future releases of `SpatialData`.

# %%
# Link table to spatial element before cropping
sdata_xenium.tables["table"].obs["region"] = "cell_boundaries"
sdata_xenium.tables["table"].obs["region"] = sdata_xenium.tables["table"].obs["region"].astype("category")
sdata_xenium.set_table_annotates_spatialelement("table", region="cell_boundaries")

# %%
sdata_xenium_crop = crop(sdata_xenium, bb_xmin, bb_ymin, bb_xmax, bb_ymax)

# %% [markdown]
# The `SpatialData` plot below shows the cell boundaries colored by cell area.
# Xenium version 1 uses nuclear expansion to generate cell boundaries.

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

# Plot DAPI image
sdata_xenium_crop.pl.render_images("morphology_focus").pl.show(
    ax=axes[0], title="DAPI image", coordinate_systems="global"
)

# Plot overlay of nuclei and cell boundaries colored by cell area
sdata_xenium_crop.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="cell_area",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=axes[1], title="Overlay of nuclei and cell masks colored by cell area", colorbar=True)

# %% [markdown]
# Finally, we can read these into SegTraQ.
# Don't worry if you do not know all of these parameters up front,
# you can start by simply passing an `sdata` object into the function and it will tell
# you exactly which parameters need to be set.

# %%
st_xenium = segtraq.SegTraQ(
    sdata_xenium,
    tables_centroid_x_key=None,
    tables_centroid_y_key=None,
    points_background_id=-1,  # "UNASSIGNED" for Xenium prime
    shapes_cell_id_key=None,
    nucleus_shapes_cell_id_key=None,
)

# %% [markdown]
# ## BIDCell

# %% [markdown]
# Depending on the segmentation method applied to re-segment the Xenium data,
# the data output format will differ. No `SpatialData` readers are available for individual segmentation methods.
# Below, we demonstrate how to load data from the output of the segmentation method `BIDCell` into `SpatialData`.
#
# The single-cell expression and metadata are first loaded into an `AnnData` object.

# %%
csv_files = list(data_path.glob("bidcell_output/cell_gene_matrices/202*/cell*.csv"))
dfs = [pd.read_csv(f) for f in csv_files]
merged_df = pd.concat(dfs, ignore_index=True).sort_values("cell_id").reset_index(drop=True)

meta_cols = ["cell_id", "cell_centroid_x", "cell_centroid_y", "cell_size"]
expr_cols = [c for c in merged_df.columns if c not in meta_cols]

obs = merged_df[meta_cols].copy()
obs["cell_id"] = obs["cell_id"].astype(np.int32)
obs["region"] = "cell_boundaries"
obs["region"] = obs["region"].astype("category")  # required for new version of SpatialData

X = merged_df[expr_cols].to_numpy()
var = pd.DataFrame(index=pd.Index(expr_cols))

adata = ad.AnnData(X=X, obs=obs, var=var)

# %% [markdown]
# `BIDCell` outputs cell and nucleus boundaries as a raster (label image) rather than vector shapes (polygons).
# Therefore, we first convert the rasterized boundaries into polygon geometries using `labels_to_shapes`.
# Nucleus boundaries are not copied from Xenium, because BIDCell, re-runs nuclear segmentation internally via Cellpose.

# %%
cell_label_files = list(data_path.glob("bidcell_output/model_outputs/202*/test_output/*_connected.tif"))
cell_labels = tiff.imread(cell_label_files[0])

nucleus_labels = tiff.imread(data_path / "bidcell_output/nuclei.tif")

cell_shapes_gdf = labels_to_shapes(cell_labels, simplify_tolerance=0.5)
nucleus_shapes_gdf = labels_to_shapes(nucleus_labels, simplify_tolerance=0.5)

# %% [markdown]
# `BIDCell` filters the Xenium transcripts file prior to segmentation by removing
# low-quality transcripts (qv < 20) and control probes.
# However, the cell IDs in the processed transcripts file (`transcripts_processed.csv`)
# still correspond to the original Xenium segmentation. Therefore,
# we reassign cell IDs based on the BIDCell segmentation masks.

# %%
transcripts_path = data_path / "bidcell_output/transcripts_processed.csv"

transcripts_df = pd.read_csv(transcripts_path, index_col=0)
transcripts_df.rename(
    columns={"cell_id": "original_cell_id", "x_location": "x", "y_location": "y", "z_location": "z"}, inplace=True
)
x = np.rint(transcripts_df["x"]).astype(int)
y = np.rint(transcripts_df["y"]).astype(int)
transcripts_df["cell_id"] = cell_labels[y, x]  # reassign BIDCell cell IDs
transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")

# %% [markdown]
# The `SpatialData` object can then be built from the `AnnData` object,
# the `cell_boundaries`, `nucleus_boundaries` and `transcripts`.
# We link the table observations to the shapes by specifying `region_key`, `region`, and `instance_key`.
# `region_key` is the column in `adata.obs` (e.g. `"region"`) that indicates which SpatialData
# element the observations refer to (here `"cell_boundaries"`).
# `instance_key` is the column in `adata.obs` (e.g. `"cell_id"`) containing the object IDs,
# which are matched to the **index** of `sdata.shapes[region]`.
# Therefore, make sure that `sdata.shapes["cell_boundaries"].index` aligns with `adata.obs[instance_key]`.

# %%
sdata_bidcell = sd.SpatialData(
    points={"transcripts": sd.models.PointsModel.parse(transcripts_df)},
    shapes={
        "cell_boundaries": sd.models.ShapesModel.parse(cell_shapes_gdf),
        "nucleus_boundaries": sd.models.ShapesModel.parse(nucleus_shapes_gdf),
    },
    tables={
        "table": sd.models.TableModel.parse(
            adata, region_key="region", region="cell_boundaries", instance_key="cell_id"
        )
    },
)

# %% [markdown]
# Finally, we read in the image. Here, we have to make sure that the individual layers
# are aligned and hence set transformations in line with the xenium data.
# If you do not have an `sdata_xenium` object that you can read the transformations from,
# you can also read in the image manually using `sd.models.Image2DModel(tiff.imread("morphology_focus.ome.tif"))`.

# %%
sdata_bidcell.images["morphology_focus"] = sdata_xenium.images["morphology_focus"]
xenium_transformation = get_transformation(sdata_xenium.shapes["cell_boundaries"])
set_transformation(sdata_bidcell.shapes["cell_boundaries"], xenium_transformation)
set_transformation(sdata_bidcell.shapes["nucleus_boundaries"], xenium_transformation)
set_transformation(sdata_bidcell.points["transcripts"], xenium_transformation)

# %% [markdown]
# We crop the `SpatialData` object as before to visualize the data.

# %%
sdata_bidcell_crop = crop(sdata_bidcell, bb_xmin, bb_ymin, bb_xmax, bb_ymax)

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

# Plot DAPI image
sdata_bidcell_crop.pl.render_images("morphology_focus").pl.show(
    ax=axes[0], title="DAPI image", coordinate_systems="global"
)

# Plot overlay of nuclei and cell boundaries colored by cell area
sdata_bidcell_crop.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="cell_size",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=axes[1], title="Overlay of nuclei and cell masks colored by cell area", colorbar=True)

# %% [markdown]
# Finally, we can initialize the `SegTraQ` object, based on which all metrics can be computed.
# Naming will differ between segmentation methods, e.g. `cell_area` in Xenium and
# `cell_size` in BIDCell in addition to other settings like the `points_background_id`.
# Please set parameters accordingly.

# %%
st_bidcell = segtraq.SegTraQ(
    sdata_bidcell,
    tables_area_key="cell_size",
    tables_centroid_x_key="cell_centroid_x",
    tables_centroid_y_key="cell_centroid_y",
    points_background_id=0,
)

# %% [markdown]
# ## Proseg 2

# %% [markdown]
# Below, we show how to read data from the older proseg version 2 into the `SpatialData` format.
# Proseg version 2 does not output a `SpatialData` object, so the latter has to be built from scratch.
#
# We first build the `AnnData` object.

# %%
counts_df = pd.read_csv(data_path / "proseg_output_v2/expected-counts.csv.gz", compression="gzip")
X = csr_matrix(np.rint(counts_df.values).astype(int, copy=False))

obs = pd.read_csv(data_path / "proseg_output_v2/cell-metadata.csv.gz", compression="gzip")
obs["region"] = "cell_boundaries"
obs["region"] = obs["region"].astype("category")  # required for new version of SpatialData
var = pd.DataFrame(index=counts_df.columns.astype(str))
adata = ad.AnnData(X=X, obs=obs, var=var)

# %% [markdown]
# Next, we read the vector shapes (polygons). These are stored in a `GeoJSON` file and can be read
# using `read_geojson_gz` and `to_cell_shapes`. Proseg is a quasi-3D method and provides
# cell boundaries for each z-layer and well as a 2D projection.

# %%
polygons_layers_gz_path = data_path / "proseg_output_v2/cell-polygons-layers.geojson.gz"
polygons_gz_path = data_path / "proseg_output_v2/cell-polygons.geojson.gz"

shapes_dict = {}

# 3D layers
polygons_layers_gdf = read_geojson_gz(polygons_layers_gz_path)
for z, gdf in polygons_layers_gdf.groupby("layer", sort=True):
    shapes_dict[f"cell_boundaries_z{int(z)}"] = sd.models.ShapesModel.parse(
        gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    )
    shapes_dict[f"cell_boundaries_z{int(z)}"].set_index("cell", drop=True, inplace=True)
# 2D projection
gdf = read_geojson_gz(polygons_gz_path)
shapes_dict["cell_boundaries"] = sd.models.ShapesModel.parse(gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty])
shapes_dict["cell_boundaries"].set_index("cell", drop=True, inplace=True)

# %% [markdown]
# Next, we load transcripts. Proseg internally filters control probes and transcripts
# with a `qv` < 20. Thus, there will be fewer transcripts than in `sdata_xenium`.
# **In Proseg, "x", "y" and "z" columns in the transcripts correspond to repositioned transcripts,
# while the "observed" columns correspond to raw input positions. For SegTraQ, we use the raw positions.**
# We will rename these columns to avoid misalignment when cropping the data via `query.bounding_box()`.

# %%
transcripts_df = pd.read_csv(data_path / "proseg_output_v2/transcript-metadata.csv.gz", compression="gzip")
transcripts_df["gene"] = transcripts_df["gene"].astype("category")
transcripts_df = transcripts_df.rename(
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
# The `SpatialData` object can then be built from the `AnnData` object, the `cell_boundaries` and `transcripts`.
# We link the table observations to the shapes by specifying `region_key`, `region`, and `instance_key`.
# `region_key` is the column in `adata.obs` (e.g. `"region"`) that indicates which SpatialData
# element the observations refer to (here `"cell_boundaries"`).
# `instance_key` is the column in `adata.obs` (e.g. `"cell_id"`) containing the object IDs,
# which are matched to the **index** of `sdata.shapes[region]`.
# Therefore, make sure that `sdata.shapes["cell_boundaries"].index` aligns with `adata.obs[instance_key]`.

# %%
sdata_proseg2 = sd.SpatialData(
    points={"transcripts": sd.models.PointsModel.parse(transcripts_df, feature_key="gene")},
    shapes=shapes_dict,
    tables={
        "table": sd.models.TableModel.parse(adata, region_key="region", region="cell_boundaries", instance_key="cell")
    },
)

# %% [markdown]
# The `images` and `nucleus_boundaries` can be copied from `sdata_xenium` since these
# are not changed by proseg segmentation. We have to make sure that the individual
# layers are aligned and hence set transformations in line with the xenium data.

# %%
sdata_proseg2.images["morphology_focus"] = sdata_xenium.images["morphology_focus"]
sdata_proseg2.shapes["nucleus_boundaries"] = sdata_xenium.shapes["nucleus_boundaries"]

xenium_transformation = get_transformation(sdata_xenium.shapes["cell_boundaries"])
set_transformation(sdata_proseg2.shapes["cell_boundaries"], xenium_transformation)

for k, shape_layer in sdata_proseg2.shapes.items():
    if k.startswith("cell_boundaries_z"):
        set_transformation(shape_layer, xenium_transformation)

set_transformation(sdata_proseg2.points["transcripts"], xenium_transformation)

# %% [markdown]
# Proseg applies clipping to the transcript z-coordinates and overwrites the original values in `observed_z`.
# To preserve the raw Xenium z-coordinates for downstream analyses, we map the original `z`
# values back to the Proseg transcript table using the unique `transcript_id`.
# See more info in this [issue](https://github.com/dcjones/proseg/issues/136).

# %%
proseg_tx = sdata_proseg2.points["transcripts"].compute()
proseg_tx = proseg_tx.reset_index(drop=True)
xenium_tx = sdata_xenium.points["transcripts"].compute()
xenium_tx = xenium_tx.reset_index(drop=True)

proseg_tx["z"] = proseg_tx["transcript_id"].map(xenium_tx.set_index("transcript_id")["z"])

transformation = sdata_proseg2.points["transcripts"].attrs["transform"]

sdata_proseg2.points["transcripts"] = PointsModel.parse(
    proseg_tx,
    coordinates={"x": "x", "y": "y", "z": "z"},
)

sdata_proseg2.points["transcripts"].attrs["transform"] = transformation

# %% [markdown]
# We crop the `SpatialData` object as before to visualize the data.

# %%
sdata_proseg2_crop = crop(sdata_proseg2, bb_xmin, bb_ymin, bb_xmax, bb_ymax)

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

# Plot DAPI image
sdata_proseg2_crop.pl.render_images("morphology_focus").pl.show(
    ax=axes[0], title="DAPI image", coordinate_systems="global"
)

# Plot overlay of nuclei and cell boundaries colored by volume (proseg measures volume instead of area)
sdata_proseg2_crop.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="volume",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=axes[1], title="Overlay of nuclei and cell masks colored by volume", colorbar=True)

# %% [markdown]
# Finally, we can initialize the `SegTraQ` object, based on which all metrics can be computed.

# %%
st_proseg2 = segtraq.SegTraQ(
    sdata_proseg2,
    shapes_cell_id_key="cell",
    nucleus_shapes_cell_id_key="segtraq_id",
    tables_cell_id_key="cell",
    points_cell_id_key="assignment",
    points_gene_key="gene",
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
    points_background_id=2**32 - 1,  # background ID used in Proseg,
    tables_area_key=None,
)

# %% [markdown]
# ## Proseg 3
#
# Proseg 3 already provides a `SpatialData` object as output, which can be read in easily.

# %%
sdata_proseg3 = sd.read_zarr(data_path / "proseg_output_v3/proseg-output.zarr")
sdata_proseg3

# %% [markdown]
# **In Proseg, "x", "y" and "z" columns in the transcripts correspond to repositioned transcripts,
# while the "observed" columns correspond to raw input positions.
# For SegTraQ, we use the raw positions.**
# We will rename these columns to avoid misalignment when cropping the data via "query.bounding_box`.

# %%
pts = sdata_proseg3.points["transcripts"]
pts = pts.rename(
    columns={
        "x": "repositioned_x",
        "y": "repositioned_y",
        "z": "repositioned_z",
        "observed_x": "x",
        "observed_y": "y",
        "observed_z": "z",
    }
)
pts["gene"] = pts["gene"].astype("category")

sdata_proseg3.points["transcripts"] = sd.models.PointsModel.parse(pts)

# %% [markdown]
# Note that this object only contains one `cell_boundaries` layer in the shapes.
# To make full use of the 3D metrics of `SegTraQ`, we can also read in the segmentations at different z layers.
#
# The table observations are linked to the shapes via `region_key`, `region`, and `instance_key`.
# `region_key` is the column in `adata.obs` (e.g. `"region"`) that indicates which
# SpatialData element the observations refer to (here `"cell_boundaries"`).
# `instance_key` is the column in `adata.obs` (here `"cell"`) containing the object IDs,
# which are matched to the **index** of `sdata.shapes[region]`.
# Therefore, make sure that `sdata.shapes["cell_boundaries"].index` aligns with `adata.obs[instance_key]`.

# %%
sdata_proseg3["table"].uns

# %%
polygons_layers_gz_path = data_path / "proseg_output_v3/cell-polygons-layers.geojson.gz"
shapes_dict = {}

polygons_layers_gdf = read_geojson_gz(polygons_layers_gz_path)
for z, gdf in polygons_layers_gdf.groupby("layer", sort=True):
    sdata_proseg3.shapes[f"cell_boundaries_z{int(z)}"] = sd.models.ShapesModel.parse(
        gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    )
    sdata_proseg3.shapes[f"cell_boundaries_z{int(z)}"].set_index(
        "cell", drop=True, inplace=True
    )  # make sure that index aligns with adata.obs[instance_key]

sdata_proseg3.shapes["cell_boundaries"].set_index(
    "cell", drop=True, inplace=True
)  # make sure that index aligns with adata.obs[instance_key]

# %% [markdown]
# The `images` and `nucleus_boundaries` can be copied from `sdata_xenium` since
# these are not changed by proseg segmentation. We have to make sure that the individual
# layers are aligned and hence set transformations in line with the xenium data.

# %%
sdata_proseg3.images["morphology_focus"] = sdata_xenium.images["morphology_focus"]
sdata_proseg3.shapes["nucleus_boundaries"] = sdata_xenium.shapes["nucleus_boundaries"]

xenium_transformation = get_transformation(sdata_xenium.shapes["cell_boundaries"])
set_transformation(sdata_proseg3.shapes["cell_boundaries"], xenium_transformation)

for k, shape_layer in sdata_proseg3.shapes.items():
    if k.startswith("cell_boundaries_z"):
        set_transformation(shape_layer, xenium_transformation)

set_transformation(sdata_proseg3.points["transcripts"], xenium_transformation)

# %% [markdown]
# Proseg applies clipping to the transcript z-coordinates and overwrites the original
# values in `observed_z`. To preserve the raw Xenium z-coordinates for downstream analyses,
# we map the original `z` values back to the Proseg transcript table using the unique `transcript_id`.
# See more info in this [issue](https://github.com/dcjones/proseg/issues/136).

# %%
proseg_tx = sdata_proseg3.points["transcripts"].compute()
proseg_tx = proseg_tx.reset_index(drop=True)
xenium_tx = sdata_xenium.points["transcripts"].compute()
xenium_tx = xenium_tx.reset_index(drop=True)

proseg_tx["z"] = proseg_tx["transcript_id"].map(xenium_tx.set_index("transcript_id")["z"])

transformation = sdata_proseg3.points["transcripts"].attrs["transform"]

sdata_proseg3.points["transcripts"] = PointsModel.parse(
    proseg_tx,
    coordinates={"x": "x", "y": "y", "z": "z"},
)

sdata_proseg3.points["transcripts"].attrs["transform"] = transformation

# %% [markdown]
# We crop the `SpatialData` object as before to visualize the data.

# %%
sdata_proseg3_crop = crop(sdata_proseg3, bb_xmin, bb_ymin, bb_xmax, bb_ymax)

# %%
axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)[1].flatten()

# Plot DAPI image
sdata_proseg3_crop.pl.render_images("morphology_focus").pl.show(
    ax=axes[0], title="DAPI image", coordinate_systems="global"
)

# Plot overlay of nuclei and cell boundaries colored by volume (proseg measures volume instead of area)
sdata_proseg3_crop.pl.render_shapes(
    element="nucleus_boundaries",
    fill_alpha=0.2,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.render_shapes(
    element="cell_boundaries",
    color="volume",
    cmap="viridis",
    fill_alpha=0.5,
    outline_alpha=1.0,
    outline_width=0.5,
    outline_color="black",
).pl.show(ax=axes[1], title="Overlay of nuclei and cell masks colored by volume", colorbar=True)

# %% [markdown]
# Finally, we can initialize the `SegTraQ` object, based on which all metrics can be computed.

# %%
filter_kwargs = {
    "min_qv": None,
    "control_prefixes": (
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword",
        "BLANK_",
        "Blank-",
        "NegPrb",
        "DeprecatedCodeword_",
        "UnassignedCodeword_",
        "Intergenic_Region_",
    ),
    "control_genes": (),
    "inplace": True,
}

st_proseg3 = segtraq.SegTraQ(
    sdata_proseg3,
    points_cell_id_key="assignment",
    points_background_id=None,
    points_gene_key="gene",
    tables_area_key=None,
    tables_cell_id_key="cell",
    shapes_cell_id_key="cell",
    nucleus_shapes_cell_id_key="segtraq_id",
    tables_centroid_x_key="centroid_x",
    tables_centroid_y_key="centroid_y",
    filter_kwargs=filter_kwargs,
)

# %% [markdown]
# ## Segger

# %% [markdown]
# **Segger is still under very active development, so output formats might change in the future.**
#
# Segger outputs single-cell data already in `AnnData` format.
# The `AnnData` object does not contain a cell ID column.
# We will add it, to be able to link the `table` to the `shapes` in the `SpatialData` object created below.

# %%
adata = ad.read_h5ad(
    data_path / "segger_output/benchmarks/segger_output_0.5_False_4_12_15_3_20260114/segger_adata.h5ad"
)
adata.obs["cell_id"] = adata.obs_names
adata.obs["region"] = "cell_boundaries"
adata.obs["region"] = adata.obs["region"].astype("category")  # required for new version of SpatialData

# %% [markdown]
# Next, we read the vector shapes (polygons). We filter out cells with invalid shapes from
# the `AnnData` object and the `shapes`. In addition, we filter out cell IDs that
# exist in the `shapes` but not in the `AnnData` object (probably due to filtering during Segger processing).
#
# The table observations are linked to the shapes via `region_key`, `region`,
# and `instance_key`. `region_key` is the column in `adata.obs` (e.g. `"region"`)
# that indicates which SpatialData element the observations refer to (here `"cell_boundaries"`).
# `instance_key` is the column in `adata.obs` (here `"cell_id"`) containing the object IDs,
# which are matched to the **index** of `sdata.shapes[region]`.
# Therefore, make sure that `sdata.shapes["cell_boundaries"].index` aligns with `adata.obs[instance_key]`.

# %%
gdf = gpd.read_parquet(
    data_path / "segger_output/benchmarks/segger_output_0.5_False_4_12_15_3_20260114/segger_boundaries.parquet"
)
gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

adata = adata[adata.obs["cell_id"].isin(gdf["cell_id"])].copy()
gdf = gdf[gdf["cell_id"].isin(adata.obs["cell_id"])]
gdf.set_index("cell_id", inplace=True, drop=True)

# %% [markdown]
# We load `segger_transcripts.parquet`, which contains transcripts assigned to cell IDs
# by segger (`segger_cell_id`). The original cell ID is renamed from `cell_id` to `original_cell_id`
# to avoid confusion. Segger applies a more stringent filtering with `min_qv = 30`
# as compared to other methods, so the number of transcripts will be lower.

# %%
transcripts = pd.read_parquet(
    data_path / "segger_output/benchmarks/segger_output_0.5_False_4_12_15_3_20260114/segger_transcripts.parquet"
)
transcripts = transcripts.rename(columns={"cell_id": "original_cell_id"})  # to avoid confusion with segger_cell_id
transcripts.rename(
    columns={"x_location": "x", "y_location": "y", "z_location": "z"}, inplace=True
)  # required for building SpatialData

# set transcripts that do not exist in adata to UNASSIGNED
transcripts.loc[~transcripts["segger_cell_id"].isin(adata.obs["cell_id"]), "segger_cell_id"] = "UNASSIGNED"
transcripts["feature_name"] = transcripts["feature_name"].str.decode("utf-8")
transcripts["feature_name"] = transcripts["feature_name"].astype("category")

# %% [markdown]
# The `SpatialData` object can then be built from the `AnnData` object, the `cell_boundaries`,
# `nucleus_boundaries` and `transcripts`. We also link the table observations to the shapes
# by setting the `region_key` and `instance_key`, as explained above.

# %%
sdata_segger = sd.SpatialData(
    points={"transcripts": sd.models.PointsModel.parse(transcripts)},
    shapes={"cell_boundaries": sd.models.ShapesModel.parse(gdf)},
    tables={
        "table": sd.models.TableModel.parse(
            adata, region_key="region", region="cell_boundaries", instance_key="cell_id"
        )
    },
)

# %% [markdown]
# The `images` and `nucleus_boundaries` can be copied from `sdata_xenium` since these
# are not changed by segger segmentation. We have to make sure that the individual
# layers are aligned and hence set transformations in line with the xenium data.

# %%
sdata_segger.images["morphology_focus"] = sdata_xenium.images["morphology_focus"]
sdata_segger.shapes["nucleus_boundaries"] = sdata_xenium.shapes["nucleus_boundaries"]

xenium_transformation = get_transformation(sdata_xenium.shapes["cell_boundaries"])
set_transformation(sdata_segger.shapes["cell_boundaries"], xenium_transformation)
set_transformation(sdata_segger.points["transcripts"], xenium_transformation)

# %% [markdown]
# We crop the `SpatialData` object as before to visualize the data.

# %%
sdata_segger_crop = crop(sdata_segger, bb_xmin, bb_ymin, bb_xmax, bb_ymax)

# %% [markdown]
# Finally, we can initialize the `SegTraQ` object, based on which all metrics can be computed.

# %%
st_segger = segtraq.SegTraQ(
    sdata_segger,
    points_cell_id_key="segger_cell_id",
    points_background_id="UNASSIGNED",
    tables_area_key="cell_area",
    tables_cell_id_key="cell_id",
    tables_centroid_x_key="cell_centroid_x",
    tables_centroid_y_key="cell_centroid_y",
    nucleus_shapes_cell_id_key="segtraq_id",
)

# %% [markdown]
# ## Session Info

# %%
print(sd.__version__)  # spatialdata
print(spatialdata_plot.__version__)
