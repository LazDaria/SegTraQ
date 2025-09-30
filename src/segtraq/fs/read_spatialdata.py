from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile as tiff
from pyarrow import parquet as pq
from rasterio.features import rasterize
from scipy.io import mmread
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.validation import make_valid
from skimage.measure import find_contours
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)

from ..fs import create_spatialdata

# -----------------------------------------------------------------------------
# Helper: build polygons from per-vertex rows
# -----------------------------------------------------------------------------


def build_cell_polygons_from_vertices(
    df: pd.DataFrame,
    id_col: str = "label_id",
    x_col: str = "vertex_x",
    y_col: str = "vertex_y",
    keep_attrs: list[str] = ("cell_id",),
    drop_closed_duplicate: bool = True,
    fix_invalid: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convert a Xenium vertex table (one row per boundary vertex) into a GeoDataFrame
    with one Polygon per unique label_id.

    Parameters
    ----------
    df : DataFrame with at least [id_col, x_col, y_col]
    id_col : identifier of the cell (matches label image IDs)
    x_col, y_col : vertex coordinates in microns
    keep_attrs : columns to carry over (first value per id is used)
    drop_closed_duplicate : drop trailing duplicate point used to close ring
    fix_invalid : attempt to repair invalid polygons with shapely.make_valid

    Returns
    -------
    GeoDataFrame with columns [id_col, keep_attrs..., geometry]
    """

    rows = []
    geoms = []

    # groupby preserves input row order; vertices are provided in clockwise order
    # (10x docs: vertices appear clockwise, first and last point duplicate to close the polygon)
    for label_id, g in df.groupby(id_col, sort=False):
        xs = g[x_col].to_numpy()
        ys = g[y_col].to_numpy()

        # drop last point if it duplicates the first (common in Xenium files)
        if drop_closed_duplicate and len(xs) >= 2 and xs[0] == xs[-1] and ys[0] == ys[-1]:
            xs = xs[:-1]
            ys = ys[:-1]

        # need at least 3 vertices
        if len(xs) < 3:
            continue

        poly = Polygon(np.column_stack([xs, ys]))

        if fix_invalid and not poly.is_valid:
            poly = make_valid(poly)
            # make_valid can return GeometryCollection; keep polygonal parts
            if poly.geom_type == "GeometryCollection":
                polys = [p for p in poly.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:  # nothing polygonal left
                    continue
                poly = polys[0] if len(polys) == 1 else MultiPolygon([p for p in polys if p.geom_type == "Polygon"])

        if poly.is_empty:
            continue

        # pick representative attributes (first row in the group)
        row = {id_col: int(label_id) if pd.notna(label_id) else None}
        for a in keep_attrs:
            if a in g.columns:
                row[a] = g[a].iloc[0]
        rows.append(row)
        geoms.append(poly)

    gdf = gpd.GeoDataFrame(rows, geometry=geoms)
    gdf.set_index(id_col, inplace=True)  # set index for labels/boundaries link

    return gdf


# -----------------------------------------------------------------------------
# Helper: mode projection across Z for labels
# -----------------------------------------------------------------------------


def labels_mode_projection(stack: np.ndarray) -> np.ndarray:
    # stack: (Z, H, W), dtype integer labels
    Z, H, W = stack.shape
    flat = stack.reshape(Z, -1).T  # (H*W, Z)
    max_id = int(stack.max())
    counts = np.zeros((flat.shape[0], max_id + 1), dtype=np.int32)
    idx = np.arange(flat.shape[0])

    # count occurrences per pixel across slices
    for z in range(Z):
        np.add.at(counts, (idx, flat[:, z]), 1)

    counts[:, 0] = 0
    winner = counts.argmax(axis=1).astype(np.int32)
    return winner.reshape(H, W)


# -----------------------------------------------------------------------------
# Helper: extract polygon boundaries from a 2D labels image
# -----------------------------------------------------------------------------


def labels_to_shapes(label_img: np.ndarray, simplify_tolerance: float | None = 0.5) -> gpd.GeoDataFrame:
    """
    Convert a 2D labels image into polygon boundaries (one polygon per instance).
    Returns a GeoDataFrame indexed by label_id with a 'geometry' column.
    """
    assert label_img.ndim == 2, "label image must be 2D"

    geoms = []
    ids = []

    # Collect unique labels (excluding background 0)
    label_ids = np.unique(label_img)
    label_ids = label_ids[label_ids != 0]

    for lid in label_ids:
        mask = (label_img == lid).astype(np.uint8)
        # find contours at level 0.5 around the binary mask
        contours = find_contours(mask, 0.5)  # list of (N,2) arrays in (row=y, col=x)
        polys = []
        for c in contours:
            # flip to (x, y); ensure polygon is valid
            poly = Polygon(np.c_[c[:, 1], c[:, 0]])
            if not poly.is_valid or poly.area == 0:
                continue
            if simplify_tolerance is not None and simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            polys.append(poly)
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        geoms.append(geom)
        ids.append(int(lid))

    gdf = gpd.GeoDataFrame({"cell_id": ids, "label_id": ids, "geometry": geoms}).set_index("label_id")
    return gdf


# -----------------------------------------------------------------------------
# Reader: 10x Xenium
# -----------------------------------------------------------------------------


def read_xenium(path_to_data: Path) -> SpatialData:
    """
    Create spatialdata object from subset 10x Xenium data.

    Parameters:
    path_to_data (Path): Path to the directory containing subset 10x Xenium data.

    Returns
    SpatialData: A spatialdata object.
    """
    # Table for sdata
    with gzip.open(path_to_data / "cell_feature_matrix" / "matrix.mtx.gz", "rt") as f:
        X = mmread(f).tocsr()  # shape: n_features x n_barcodes

    features = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "features.tsv.gz", sep="\t", header=None, compression="gzip"
    )

    barcodes = pd.read_csv(
        path_to_data / "cell_feature_matrix" / "barcodes.tsv.gz", sep="\t", header=None, compression="gzip"
    )[0]

    gene_mask = features[2] == "Gene Expression"
    X = X[gene_mask.values, :]  # subset ROWS (features)
    var = pd.DataFrame(index=features.loc[gene_mask, 1].astype(str).values)
    var.index.name = "gene_symbol"

    meta_df = pd.read_csv(path_to_data / "cells.csv.gz", compression="gzip")
    rename_map = {
        "cell_centroid_x": "x_centroid",
        "cell_centroid_y": "y_centroid",
        "cell_area": "cell_area",
    }
    for k, v in list(rename_map.items()):
        if k in meta_df.columns and v not in meta_df.columns:
            meta_df[v] = meta_df[k]

    meta_df = meta_df.set_index("cell_id", drop=False)
    common = barcodes[barcodes.isin(meta_df.index)]
    obs = meta_df.loc[common].copy()
    obs.reset_index(drop=True, inplace=True)
    X = X[:, barcodes.isin(meta_df.index).to_numpy()]

    obs["label_id"] = np.arange(1, len(obs) + 1, dtype=np.int32)
    obs["region"] = pd.Categorical(["cell_labels"] * len(obs))

    adata = ad.AnnData(X=X.T, obs=obs, var=var)

    # Image for sdata
    dapi = tiff.imread(path_to_data / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]

    # Label for sdata
    cell_labels = tiff.imread(path_to_data / "cell_mask_um.tif")
    nucleus_labels = tiff.imread(path_to_data / "nuc_mask_um.tif")

    # Shapes for sdata
    cell_shapes = pd.read_parquet(path_to_data / "cell_boundaries.parquet")
    cell_shapes_gpd = build_cell_polygons_from_vertices(cell_shapes)

    nucleus_shapes = pd.read_parquet(path_to_data / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)

    # Points for sdata
    transcripts_df = pq.read_table(path_to_data / "transcripts.parquet").to_pandas()
    transcripts_df["feature_name"] = transcripts_df["feature_name"].astype("category")
    transcripts_df["is_gene"] = transcripts_df["is_gene"].astype("str")
    transcripts_df = transcripts_df.rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})

    # --- assemble SpatialData
    sdata_object = create_spatialdata(
        points=transcripts_df,
        labels={"cell_labels": cell_labels, "nucleus_labels": nucleus_labels},
        shapes={"cell_boundaries": cell_shapes_gpd, "nucleus_boundaries": nucleus_shapes_gpd},
        tables=adata,
        images=dapi,
    )
    return sdata_object


# -----------------------------------------------------------------------------
# Reader: Proseg 2.0
# -----------------------------------------------------------------------------


def read_proseg_2(path_to_proseg_data: Path, path_to_10xdata: Path) -> SpatialData:
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
    table_sd = TableModel.parse(
        adata,
        region_key="region",
        region="cell_labels",
        instance_key="label_id",
    )

    # ---------------------------
    # Image (DAPI image from 10x)
    # ---------------------------
    dapi = tiff.imread(path_to_10xdata / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]
    image_sd = Image2DModel.parse(dapi, scale_factors=(2, 2, 2), dims=["c", "y", "x"])

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
    # stack = np.zeros((len(z_levels), H, W), dtype=np.uint32)
    # max_proj = np.zeros((H, W), dtype=np.uint32)

    for _zi, z in enumerate(z_levels):
        layer_gdf = gdf[gdf["layer"] == z]

        # Shapes (index must be instance ids to join with table.instance_key)
        layer_shapes = layer_gdf.set_index("cell")["geometry"].to_frame().copy()
        layer_shapes.index.name = "label_id"
        layer_shapes.index += 1
        layer_shapes["cell_id"] = layer_shapes.index
        shapes_dict[f"cell_boundaries_z{int(z)}"] = ShapesModel.parse(layer_shapes)

        # Labels via rasterize (value == cell id, background 0)
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)

        labels_dict[f"cell_labels_z{int(z)}"] = Labels2DModel.parse(img, scale_factors=(2, 2, 2), dims=["y", "x"])

        # stack[zi] = img
        # max_proj = np.maximum(max_proj, img)
    # proj = labels_mode_projection(stack)  # more representative than MIP

    # MIP label (2D)
    # labels_dict["cell_labels"] = Labels2DModel.parse(
    #     proj, scale_factors=(2, 2, 2), dims=["y", "x"]
    # )

    # # 3D label stack: (z, y, x)
    # labels_dict["cell_labels_3d"] = Labels3DModel.parse(
    #     stack,
    #     scale_factors=None,
    #     dims=["z", "y", "x"]
    # )

    # # Generate 2D cell_boundaries from label projection
    # polys2d = labels_to_shapes(proj, simplify_tolerance=0.5)

    # shapes_dict["cell_boundaries"] = ShapesModel.parse(polys2d)

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)
    shapes_dict["nucleus_boundaries"] = ShapesModel.parse(nucleus_shapes_gpd)

    # -------------------------
    # Nuc labels (from 10x)
    # -------------------------

    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = Labels2DModel.parse(nucleus_labels, scale_factors=(2, 2, 2), dims=["y", "x"])

    # -------------------------
    # Points (transcripts)
    # -------------------------
    tx = pd.read_csv(path_to_proseg_data / "transcript-metadata.csv.gz", compression="gzip")
    tx = tx.rename(columns={"gene": "feature_name", "assignment": "cell_id"})

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
        if c in tx.columns
    ]
    tx = tx[keep_cols].copy()

    tx["cell_id"] = (tx["cell_id"] + 1).astype(int)
    tx["feature_name"] = tx["feature_name"].astype("category")

    tx.loc[tx["cell_id"] == 2**32, "cell_id"] = 0  # uint32_max placeholder meaning “no assignment / background”

    transcripts_sd = PointsModel.parse(tx)

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = SpatialData(
        images={"morphology_focus": image_sd},
        labels=labels_dict,
        shapes=shapes_dict,
        points={"transcripts": transcripts_sd},
        tables={"table": table_sd},
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: Proseg 3.0
# -----------------------------------------------------------------------------


def read_proseg_3(path_to_proseg_data: Path, path_to_10xdata: Path) -> SpatialData:
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
    table_sd = TableModel.parse(
        adata,
        region_key="region",
        region="cell_labels",
        instance_key="label_id",
    )

    # ---------------------------
    # Image (DAPI image from 10x)
    # ---------------------------
    dapi = tiff.imread(path_to_10xdata / "dapi_um.tif")
    if dapi.ndim == 2:  # add channel axis if single-channel
        dapi = dapi[None, ...]
    image_sd = Image2DModel.parse(dapi, scale_factors=(2, 2, 2), dims=["c", "y", "x"])

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
        shapes_dict[f"cell_boundaries_z{int(z)}"] = ShapesModel.parse(layer_shapes)

        # Labels via rasterize (value == cell id, background 0)
        shapes_iter = (
            (mapping(geom), int(cid) + 1) for cid, geom in zip(layer_gdf["cell"], layer_gdf.geometry, strict=False)
        )
        img = rasterize(shapes_iter, out_shape=(H, W), fill=0, dtype=np.uint32)

        labels_dict[f"cell_labels_z{int(z)}"] = Labels2DModel.parse(img, scale_factors=(2, 2, 2), dims=["y", "x"])

        stack[zi] = img
        # max_proj = np.maximum(max_proj, img)
    proj = labels_mode_projection(stack)  # more representative than MIP

    # MIP label (2D)
    labels_dict["cell_labels"] = Labels2DModel.parse(proj, scale_factors=(2, 2, 2), dims=["y", "x"])

    # 3D label stack: (z, y, x)
    labels_dict["cell_labels_3d"] = Labels3DModel.parse(stack, scale_factors=None, dims=["z", "y", "x"])

    # Generate 2D cell_boundaries from label projection
    polys2d = labels_to_shapes(proj, simplify_tolerance=0.5)

    shapes_dict["cell_boundaries"] = ShapesModel.parse(polys2d)

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)
    shapes_dict["nucleus_boundaries"] = ShapesModel.parse(nucleus_shapes_gpd)

    # -------------------------
    # Nuc labels (from 10x)
    # -------------------------

    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    labels_dict["nucleus_labels"] = Labels2DModel.parse(nucleus_labels, scale_factors=(2, 2, 2), dims=["y", "x"])

    # -------------------------
    # Points (transcripts)
    # -------------------------
    tx = pd.read_csv(path_to_proseg_data / "transcript-metadata.csv.gz", compression="gzip")
    tx = tx.rename(columns={"gene": "feature_name", "assignment": "cell_id"})
    keep_cols = [c for c in ["transcript_id", "x", "y", "z", "feature_name", "cell_id"] if c in tx.columns]
    tx = tx[keep_cols].copy()
    tx["cell_id"] = tx["cell_id"].fillna(0)
    tx["cell_id"] = (tx["cell_id"] + 1).astype(int)
    tx["feature_name"] = tx["feature_name"].astype("category")
    transcripts_sd = PointsModel.parse(tx)

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = SpatialData(
        images={"morphology_focus": image_sd},
        labels=labels_dict,
        shapes=shapes_dict,
        points={"transcripts": transcripts_sd},
        tables={"table": table_sd},
    )
    return sdata


# -----------------------------------------------------------------------------
# Reader: BIDCell
# -----------------------------------------------------------------------------


def read_bidcell(path_to_data: Path) -> SpatialData:
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

    table_sd = TableModel.parse(
        adata,
        region_key="region",
        region="cell_labels",
        instance_key="label_id",
    )

    # Labels for sdata
    cell_labels_path = list(bidcell_path.glob("model_outputs/202*/test_output/epoch_4_step_100_connected.tif"))
    cell_labels = tiff.imread(cell_labels_path[0])
    cell_labels_sd = Labels2DModel.parse(cell_labels, scale_factors=(2, 2, 2), dims=["y", "x"])

    nucleus_labels = tiff.imread(bidcell_path / "nuclei.tif")
    nucleus_labels_sd = Labels2DModel.parse(nucleus_labels, scale_factors=(2, 2, 2), dims=["y", "x"])

    # Shapes for sdata
    cell_shapes_gdf = labels_to_shapes(cell_labels, simplify_tolerance=0.5)
    nucleus_shapes_gdf = labels_to_shapes(nucleus_labels, simplify_tolerance=0.5)

    cell_shapes_sd = ShapesModel.parse(cell_shapes_gdf)
    nucleus_shapes_sd = ShapesModel.parse(nucleus_shapes_gdf)

    dapi = tiff.imread(bidcell_path / "dapi_resized.tif")

    if dapi.ndim == 2:
        dapi = dapi[None, ...]  # (c, y, x)
    else:
        raise ValueError("Unexpected DAPI image ndim; expected 2D or 3D")

    image_sd = Image2DModel.parse(dapi, scale_factors=(2, 2, 2), dims=["c", "y", "x"])

    # Points for sdata
    transcripts = pd.read_csv(bidcell_path / "transcripts_processed.csv", index_col=0)
    # Round to pixel grid (as provided)
    x = np.rint(transcripts["x_location"]).astype(int)
    y = np.rint(transcripts["y_location"]).astype(int)

    transcripts = transcripts.copy()
    transcripts["cell_id"] = cell_labels[y, x]

    transcripts = transcripts.rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})

    transcripts["feature_name"] = transcripts["feature_name"].astype("category")
    transcripts["is_gene"] = transcripts["is_gene"].astype("string")

    transcripts_sd = PointsModel.parse(transcripts)

    # assemble spatialdata
    sdata = SpatialData(
        images={"morphology_focus": image_sd},
        labels={"cell_labels": cell_labels_sd, "nucleus_labels": nucleus_labels_sd},
        shapes={"cell_boundaries": cell_shapes_sd, "nucleus_boundaries": nucleus_shapes_sd},
        points={"transcripts": transcripts_sd},
        tables={"table": table_sd},
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
    image_sd = Image2DModel.parse(dapi, scale_factors=(2, 2, 2), dims=["c", "y", "x"])

    # -------------------------
    # Boundaries from transcripts
    # -------------------------
    boundaries_gdf = gpd.read_parquet(path_to_data / "segger_boundaries.parquet")

    gdf = boundaries_gdf[boundaries_gdf.geometry.notnull()].copy()

    unique_ids = gdf["cell_id"].unique()
    id_str_to_int = {cell_id: i + 1 for i, cell_id in enumerate(unique_ids)}

    gdf["label_id"] = gdf["cell_id"].map(id_str_to_int)
    gdf = gdf.drop(columns=["length"])

    shapes_gdf = gdf.set_index("label_id").copy()
    # There are shapes that are not present in the adata object - is segger prefiltering?
    shapes_gdf = shapes_gdf[shapes_gdf["cell_id"].isin(adata.obs["cell_id"])]
    cell_shapes_sd = ShapesModel.parse(shapes_gdf)

    # nucleus boundaries from 10x
    nucleus_shapes = pd.read_parquet(path_to_10xdata / "nucleus_boundaries.parquet")
    nucleus_shapes_gpd = build_cell_polygons_from_vertices(nucleus_shapes)
    nucleus_shapes_sd = ShapesModel.parse(nucleus_shapes_gpd)

    # -------------------------
    # Rasterize to 2D label image
    # -------------------------
    H = dapi.shape[1]
    W = dapi.shape[2]

    shapes_iter = ((mapping(geom), int(cid)) for cid, geom in zip(shapes_gdf.index, shapes_gdf.geometry, strict=False))
    label_img = rasterize(
        shapes_iter,
        out_shape=(H, W),
        fill=0,
        dtype=np.uint32,
    )
    # fewer labels than shapes - why?
    cell_labels_sd = Labels2DModel.parse(label_img, scale_factors=(2, 2, 2), dims=["y", "x"])

    # nucleus labels from 10x
    nucleus_labels = tiff.imread(path_to_10xdata / "nuc_mask_um.tif")
    nucleus_labels_sd = Labels2DModel.parse(nucleus_labels, scale_factors=(2, 2, 2), dims=["y", "x"])

    # -------------------------
    # Points from transcripts
    # -------------------------
    transcripts = pd.read_parquet(path_to_data / "segger_transcripts.parquet")
    transcripts.drop(columns=["score", "bound", "cell_id"], inplace=True)
    transcripts = transcripts.rename(
        columns={"x_location": "x", "y_location": "y", "z_location": "z", "segger_cell_id": "cell_id"}
    )

    transcripts["feature_name"] = transcripts["feature_name"].astype("category")
    transcripts["is_gene"] = transcripts["is_gene"].astype("string")
    transcripts[transcripts["cell_id"].isin(adata.obs["cell_id"])]
    # there are cells in the transcripts that are not present in the boundaries - check why - invalid shapes?
    transcripts = transcripts[transcripts["cell_id"].isin(adata.obs["cell_id"])]

    transcripts_sd = PointsModel.parse(transcripts)

    # -------------------------
    # Finalize table metadata now that we know label ids
    # -------------------------

    adata.obs["label_id"] = adata.obs["cell_id"].map(id_str_to_int)
    adata.obs["region"] = pd.Categorical(["cell_labels"] * adata.n_obs)

    table_sd = TableModel.parse(
        adata,
        region_key="region",
        region="cell_labels",
        instance_key="label_id",
    )

    # -------------------------
    # Assemble SpatialData
    # -------------------------
    sdata = SpatialData(
        images={"morphology_focus": image_sd},
        labels={"cell_labels": cell_labels_sd, "nucleus_labels": nucleus_labels_sd},
        shapes={"cell_boundaries": cell_shapes_sd, "nucleus_boundaries": nucleus_shapes_sd},
        points={"transcripts": transcripts_sd},
        tables={"table": table_sd},
    )

    return sdata
