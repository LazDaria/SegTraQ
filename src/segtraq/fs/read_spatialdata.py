from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile as tiff
from rasterio.features import rasterize
from scipy.io import mmread
from scipy.sparse import csr_matrix
from shapely.geometry import mapping
from spatialdata import SpatialData

from ..fs import create_spatialdata
from .utils import (
    build_cell_polygons_from_vertices,
    labels_mode_projection,
    labels_to_shapes,
    make_adata,
    read_dapi_image,
    read_labels,
    read_shapes,
    read_transcripts,
)


# -----------------------------------------------------------------------------
# Reader: 10x Xenium
# -----------------------------------------------------------------------------
def read_xenium(path_to_data: Path) -> SpatialData:
    """
    Read 10x Xenium data and assemble a SpatialData object.

    Parameters
    ----------
    path_to_data : Path
        Path to the directory containing subset 10x Xenium data.

    Returns
    -------
    SpatialData
        SpatialData object containing transcripts, shapes, labels, tables, and DAPI image.
    """
    # Table
    with gzip.open(path_to_data / "cell_feature_matrix" / "matrix.mtx.gz", "rt") as f:
        X = mmread(f).tocsr()
    features = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "features.tsv.gz", sep="\t", header=None, compression="gzip"
    )
    barcodes = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "barcodes.tsv.gz", sep="\t", header=None, compression="gzip"
    )[0]

    gene_mask = features[2] == "Gene Expression"
    X = X[gene_mask.values, :]
    var = pd.DataFrame(index=features.loc[gene_mask, 1].astype(str).values)
    var.index.name = "gene_symbol"

    meta_df = pd.read_csv(path_to_data / "cells.csv.gz", compression="gzip")
    rename_map = {"cell_centroid_x": "x_centroid", "cell_centroid_y": "y_centroid", "cell_area": "cell_area"}
    for k, v in rename_map.items():
        if k in meta_df.columns and v not in meta_df.columns:
            meta_df[v] = meta_df[k]

    meta_df = meta_df.set_index("cell_id", drop=False)
    common = barcodes[barcodes.isin(meta_df.index)]
    obs = meta_df.loc[common].copy()
    obs.reset_index(drop=True, inplace=True)
    X = X[:, barcodes.isin(meta_df.index).to_numpy()]

    obs["label_id"] = pd.RangeIndex(start=1, stop=len(obs) + 1, dtype=int)
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = make_adata(X.T, obs, var)

    # Image
    dapi = read_dapi_image(path_to_data / "dapi_um.tif")

    # Labels
    labels = read_labels(path_to_data / "cell_mask_um.tif", path_to_data / "nuc_mask_um.tif")

    # Shapes
    shapes = {
        "cell_boundaries": read_shapes(path_to_data / "cell_boundaries.parquet"),
        "nucleus_boundaries": read_shapes(path_to_data / "nucleus_boundaries.parquet"),
    }

    # Points
    transcripts = read_transcripts(
        path_to_data / "transcripts.parquet", rename_map={"x_location": "x", "y_location": "y", "z_location": "z"}
    )

    sdata = create_spatialdata(
        points=transcripts,
        labels=labels,
        shapes=shapes,
        tables=adata,
        images=dapi,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Proseg 2.0
# -----------------------------------------------------------------------------
def read_proseg_2(path_to_proseg_data: Path, path_to_10xdata: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from Proseg outputs.

    Assumes the following files & columns exist:
      expected-counts.csv.gz            -> cells x genes
      cell-metadata.csv.gz               -> 'cell_id','cell_centroid_x','cell_centroid_y','cell_area'
      cell-polygons-layers.geojson(.gz)  -> 'cell','layer','geometry'
      transcript-metadata.csv.gz         -> 'x_location','y_location','z_location','gene'  (optional 'cell')
    """

    # -------------------------
    # Table (counts + metadata)
    # -------------------------
    counts_df = pd.read_csv(path_to_proseg_data / "expected-counts.csv.gz", compression="gzip")
    gene_cols = counts_df.columns.astype(str)
    var = pd.DataFrame(index=gene_cols)
    var.index.name = "gene_symbol"
    X_est = counts_df.values
    X_est = csr_matrix(X_est)

    # Round to nearest int and convert to CSR
    X = np.rint(counts_df.values).astype(np.int32, copy=False)
    X = csr_matrix(X)

    obs = pd.read_csv(path_to_proseg_data / "cell-metadata.csv.gz", compression="gzip")
    obs.drop(columns=["cluster", "scale", "original_cell_id", "population", "fov"], inplace=True)
    obs.rename(columns={"cell": "cell_id"}, inplace=True)
    obs["cell_id"] = obs["cell_id"] + 1
    obs["label_id"] = obs["cell_id"]
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["X_estimated"] = X_est

    # ---------------------------
    # Image (DAPI image from 10x)
    # ---------------------------
    dapi = tiff.imread(path_to_10xdata / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]

    # -------------------------
    # Polygons -> per-layer shapes + raster labels (2D per z) + MIP + 3D
    # -------------------------
    gz_path = path_to_proseg_data / "cell-polygons-layers.geojson.gz"
    json_path = path_to_proseg_data / "cell-polygons-layers.geojson"
    if gz_path.exists() and not json_path.exists():
        with gzip.open(gz_path, "rt") as f_in, open(json_path, "w") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gdf = gpd.read_file(json_path if json_path.exists() else gz_path)
    # Expect columns: 'cell', 'layer', 'geometry'
    gdf["cell"] = gdf["cell"].astype(int)
    gdf["layer"] = gdf["layer"].astype(int)

    # Canvas from dapi image
    H = dapi.shape[1]
    W = dapi.shape[2]

    labels_dict = {}
    shapes_dict = {}

    # Per-layer shapes and labels + accumulate for 3D stack & MIP
    z_levels = sorted(gdf["layer"].unique())
    stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)
    max_proj = np.zeros((H, W), dtype=np.uint32)

    for zi, z in enumerate(z_levels):
        layer_gdf = gdf[gdf["layer"] == z]

        # Shapes (index must be instance ids to join with table.instance_key)
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = layer_shapes

        # Labels via rasterize (value == cell id, background 0)
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)

        labels_dict[f"cell_labels_z{int(z)}"] = img

        stack[zi] = img
        max_proj = np.maximum(max_proj, img)
    proj = labels_mode_projection(stack)  # more representative than MIP

    # MIP label (2D)
    labels_dict["cell_labels"] = proj

    # Generate 2D cell_boundaries from label projection
    polys2d = labels_to_shapes(proj, simplify_tolerance=0.5)
    shapes_dict["cell_boundaries"] = polys2d

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)
    shapes_dict["nucleus_boundaries"] = nucleus_shapes_gpd

    # -------------------------
    # Nuc labels (from 10x)
    # -------------------------

    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = nucleus_labels

    # -------------------------
    # Points (transcripts)
    # -------------------------
    transcripts_df = pd.read_csv(path_to_proseg_data / "transcript-metadata.csv.gz", compression="gzip")
    transcripts_df = transcripts_df.rename(columns={"gene": "feature_name", "assignment": "cell_id"})

    keep_cols = [
        c
        for c in [
            "transcript_id",
            "x",
            "y",
            "z",
            "observed_x",
            "observed_y",
            "observed_z",
            "feature_name",
            "cell_id",
            "qv",
            "probability",
        ]
        if c in transcripts_df.columns
    ]
    transcripts_df = transcripts_df[keep_cols].copy()

    transcripts_df["cell_id"] = (transcripts_df["cell_id"] + 1).astype(int)
    transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")

    transcripts_df.loc[transcripts_df["cell_id"] == 2**32, "cell_id"] = (
        0  # uint32_max placeholder meaning “no assignment / background”
    )

    sdata = create_spatialdata(
        points=transcripts_df,
        labels=labels_dict,
        shapes=shapes_dict,
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Proseg 3.0
# -----------------------------------------------------------------------------


def read_proseg_3(path_to_proseg_data: Path, path_to_10xdata: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Build a SpatialData object from Proseg outputs.

    Assumes the following files & columns exist:
      counts.mtx.gz                      -> cells x genes (Matrix Market)
      gene-metadata.csv.gz               -> 'gene_symbol'
      cell-metadata.csv.gz               -> 'cell_id','cell_centroid_x','cell_centroid_y','cell_area'
      cell-polygons-layers.geojson(.gz)  -> 'cell','layer','geometry'
      transcript-metadata.csv.gz         -> 'x_location','y_location','z_location','gene'  (optional 'cell')
    """

    # -------------------------
    # Table (counts + metadata)
    # -------------------------
    with gzip.open(path_to_proseg_data / "counts.mtx.gz", "rt") as f:
        X = mmread(f).tocsr()  # cells x genes

    X = X.astype(np.int32)

    var_df = pd.read_csv(path_to_proseg_data / "gene-metadata.csv.gz", compression="gzip")
    var = pd.DataFrame(index=var_df["gene"].astype(str).values)
    var.index.name = "gene_symbol"

    obs = pd.read_csv(path_to_proseg_data / "cell-metadata.csv.gz", compression="gzip")
    obs.drop(columns=["cluster", "scale", "original_cell_id"], inplace=True)
    obs.rename(
        columns={
            "cell": "cell_id",
            "centroid_x": "centroid_x",
            "cell_centroid_y": "centroid_y",
            "surface_area": "cell_area",
        },
        inplace=True,
    )
    obs["cell_id"] = obs["cell_id"] + 1
    obs["label_id"] = obs["cell_id"]
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # ---------------------------
    # Image (DAPI image from 10x)
    # ---------------------------
    dapi = tiff.imread(path_to_10xdata / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]

    # -------------------------
    # Polygons -> per-layer shapes + raster labels (2D per z) + MIP + 3D
    # -------------------------
    gz_path = path_to_proseg_data / "cell-polygons-layers.geojson.gz"
    json_path = path_to_proseg_data / "cell-polygons-layers.geojson"
    if gz_path.exists() and not json_path.exists():
        with gzip.open(gz_path, "rt") as f_in, open(json_path, "w") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gdf = gpd.read_file(json_path if json_path.exists() else gz_path)
    # Expect columns: 'cell', 'layer', 'geometry'
    gdf["cell"] = gdf["cell"].astype(int)
    gdf["layer"] = gdf["layer"].astype(int)

    # Canvas from dapi image
    H = dapi.shape[1]
    W = dapi.shape[2]

    labels_dict = {}
    shapes_dict = {}

    # Per-layer shapes and labels + accumulate for 3D stack & MIP
    z_levels = sorted(gdf["layer"].unique())
    stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)
    # max_proj = np.zeros((H, W), dtype=np.uint32)

    for zi, z in enumerate(z_levels):
        layer_gdf = gdf[gdf["layer"] == z]

        # Shapes (index must be instance ids to join with table.instance_key)
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = layer_shapes

        # Labels via rasterize (value == cell id, background 0)
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)

        labels_dict[f"cell_labels_z{int(z)}"] = img

        stack[zi] = img
        # max_proj = np.maximum(max_proj, img)
    proj = labels_mode_projection(stack)  # more representative than MIP

    # MIP label (2D)
    labels_dict["cell_labels"] = proj

    # Generate 2D cell_boundaries from label projection
    polys2d = labels_to_shapes(proj, simplify_tolerance=0.5)
    shapes_dict["cell_boundaries"] = polys2d

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)
    shapes_dict["nucleus_boundaries"] = nucleus_shapes_gpd

    # -------------------------
    # Nuc labels (from 10x)
    # -------------------------
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = nucleus_labels

    # -------------------------
    # Points (transcripts)
    # -------------------------
    transcripts_df = pd.read_csv(path_to_proseg_data / "transcript-metadata.csv.gz", compression="gzip")
    transcripts_df = transcripts_df.rename(columns={"gene": "feature_name", "assignment": "cell_id"})
    keep_cols = [c for c in ["transcript_id", "x", "y", "z", "feature_name", "cell_id"] if c in transcripts_df.columns]
    transcripts_df = transcripts_df[keep_cols].copy()
    transcripts_df["cell_id"] = transcripts_df["cell_id"].fillna(0)
    transcripts_df["cell_id"] = (transcripts_df["cell_id"] + 1).astype(int)
    transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts_df,
        labels=labels_dict,
        shapes=shapes_dict,
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: BIDCell
# -----------------------------------------------------------------------------


def read_bidcell(path_to_data: Path, consolidate_shapes: bool = True) -> SpatialData:
    """
    Create spatialdata object from subset BIDCell data.

    Parameters:
    path_to_data (Path): Path to the directory containing subset BIDCell data.

    Returns
    SpatialData: A spatialdata object.
    """

    bidcell_path = path_to_data

    # Table for sdata
    all_files = list(bidcell_path.glob("cell_gene_matrices/202*/cell*.csv"))
    if len(all_files) == 0:
        raise FileNotFoundError("No CSVs found under cell_gene_matrices/202*/cell*.csv")

    dfs = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sort_values("cell_id").reset_index(drop=True)
    merged_df["cell_id"] = merged_df["cell_id"].astype(int)
    merged_df = merged_df.rename(
        columns={
            "cell_size": "cell_area",
            "cell_centroid_x": "centroid_x",
            "cell_centroid_y": "centroid_y",
        }
    )

    meta_cols = ["cell_id", "centroid_x", "centroid_y", "cell_area"]

    expr_cols = [c for c in merged_df.columns if c not in meta_cols]

    obs = merged_df[meta_cols].copy()

    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))
    obs["label_id"] = obs["cell_id"].astype(np.int32)

    var = pd.DataFrame(index=pd.Index(expr_cols, name="gene_symbol"))

    from scipy import sparse  # keep local to avoid altering behavior elsewhere

    X = merged_df[expr_cols].to_numpy()
    X = sparse.csr_matrix(X)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Labels for sdata
    cell_labels_path = list(bidcell_path.glob("model_outputs/202*/test_output/epoch_4_step_100_connected.tif"))
    cell_labels = tiff.imread(cell_labels_path[0])
    nucleus_labels = tiff.imread(bidcell_path / "nuclei.tif")

    # Shapes for sdata
    cell_shapes_gdf = labels_to_shapes(cell_labels, simplify_tolerance=0.5)
    nucleus_shapes_gdf = labels_to_shapes(nucleus_labels, simplify_tolerance=0.5)

    dapi = tiff.imread(bidcell_path / "dapi_resized.tif")

    if dapi.ndim == 2:
        dapi = dapi[None, ...]  # (c, y, x)
    else:
        raise ValueError("Unexpected DAPI image ndim; expected 2D or 3D")

    # Points for sdata
    transcripts_df = pd.read_csv(bidcell_path / "transcripts_processed.csv", index_col=0)
    # Round to pixel grid (as provided)
    x = np.rint(transcripts_df["x_location"]).astype(int)
    y = np.rint(transcripts_df["y_location"]).astype(int)

    transcripts_df = transcripts_df.copy()
    transcripts_df["cell_id"] = cell_labels[y, x]

    transcripts_df = transcripts_df.rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})

    transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")
    transcripts_df["is_gene"] = transcripts_df["is_gene"].astype("string")

    # assemble spatialdata
    sdata = create_spatialdata(
        points=transcripts_df,
        labels={"cell_labels": cell_labels, "nucleus_labels": nucleus_labels},
        shapes={"cell_boundaries": cell_shapes_gdf, "nucleus_boundaries": nucleus_shapes_gdf},
        tables=adata,
        images=dapi,
        background_cell_id=0,
        consolidate_shapes=consolidate_shapes,
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Segger
# -----------------------------------------------------------------------------


def read_segger(path_to_data: Path, path_to_10xdata: Path) -> SpatialData:
    """
    Create spatialdata object from subset Segger data.

    Parameters:
    path_to_data (Path): Path to the directory containing subset Segger data.

    Returns
    SpatialData: A spatialdata object.
    """
    # -------------------------
    # Table (AnnData)
    # -------------------------
    adata = ad.read_h5ad(path_to_data / "segger_adata.h5ad")
    adata.obs.index.name = "cell_id"
    adata.obs.reset_index(inplace=True)

    adata.obs.sort_values("cell_id", ascending=True, inplace=True)
    adata.obs.reset_index(drop=True, inplace=True)
    adata = adata[adata.obs.index, :].copy()

    adata.obs.drop(columns=["transcripts", "unique_transcripts"], inplace=True)

    # ---------------------------
    # Image (DAPI image from 10x)
    # ---------------------------
    dapi = tiff.imread(path_to_10xdata / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]

    # -------------------------
    # Boundaries from transcripts
    # -------------------------
    boundaries_gdf = gpd.read_parquet(path_to_data / "segger_boundaries.parquet")

    gdf = boundaries_gdf[boundaries_gdf.geometry.notnull()].copy()

    unique_ids = gdf["cell_id"].unique()
    id_str_to_int = {cell_id: i + 1 for i, cell_id in enumerate(unique_ids)}

    gdf["label_id"] = gdf["cell_id"].map(id_str_to_int)
    gdf = gdf.drop(columns=["length"])

    cell_shapes_gdf = gdf.set_index("label_id").copy()
    # There are shapes that are not present in the adata object - is segger prefiltering?
    cell_shapes_gdf = cell_shapes_gdf[cell_shapes_gdf["cell_id"].isin(adata.obs["cell_id"])]

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gdf = build_cell_polygons_from_vertices(nucleus_shapes)

    # -------------------------
    # Rasterize to 2D label image
    # -------------------------
    H = dapi.shape[1]
    W = dapi.shape[2]

    cell_shapes_iter = (
        (mapping(geom), int(cid)) for cid, geom in zip(cell_shapes_gdf.index, cell_shapes_gdf.geometry, strict=False)
    )
    cell_labels = rasterize(
        cell_shapes_iter,
        out_shape=(H, W),
        fill=0,
        dtype=np.uint32,
    )

    # nucleus labels from 10x
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")

    # -------------------------
    # Points from transcripts
    # -------------------------
    transcripts_df = pd.read_parquet(path_to_data / "segger_transcripts.parquet")
    transcripts_df.drop(columns=["score", "bound", "cell_id"], inplace=True)
    transcripts_df = transcripts_df.rename(
        columns={"x_location": "x", "y_location": "y", "z_location": "z", "segger_cell_id": "cell_id"}
    )

    transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")
    transcripts_df["is_gene"] = transcripts_df["is_gene"].astype("string")
    transcripts_df = transcripts_df[transcripts_df["cell_id"].isin(adata.obs["cell_id"])]
    # there are cells in the transcripts that are not present in the boundaries - check why - invalid shapes?
    transcripts_df = transcripts_df[transcripts_df["cell_id"].isin(adata.obs["cell_id"])]

    # -------------------------
    # Finalize table metadata now that we know label ids
    # -------------------------

    adata.obs["label_id"] = adata.obs["cell_id"].map(id_str_to_int)
    adata.obs["region"] = pd.Categorical(["cell_labels"] * adata.n_obs)

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = create_spatialdata(
        points=transcripts_df,
        labels={"cell_labels": cell_labels, "nucleus_labels": nucleus_labels},
        shapes={"cell_boundaries": cell_shapes_gdf, "nucleus_boundaries": nucleus_shapes_gdf},
        tables=adata,
        images=dapi,
    )

    return sdata
