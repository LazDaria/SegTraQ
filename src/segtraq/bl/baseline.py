import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from shapely.geometry import MultiPolygon, Polygon

from ..utils import merge_into_obs
from .utils import count_polygons


def num_cells(sdata: sd.SpatialData, tables_key: str = "table", inplace: bool = True) -> int:
    """
    Counts the number of cells in the given SpatialData object based on the specified table key.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial information and a table.
    tables_key : str, optional
        The key in the `tables` attribute of `sdata` that corresponds to table.
        Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The number of cells found under the specified table key.
    """
    num_cells = len(sdata.tables[tables_key])
    if inplace:
        sdata.tables[tables_key].uns["num_cells"] = num_cells
    return num_cells


def num_transcripts(
    sdata: sd.SpatialData, points_key: str = "transcripts", tables_key: str = "table", inplace: bool = True
) -> int:
    """
    Counts the total number of transcripts in the given SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The total number of transcripts in the specified SpatialData object.
    """
    num_transcripts = sdata.points[points_key].shape[0].compute()
    if inplace:
        sdata.tables[tables_key].uns["num_transcripts"] = num_transcripts

    return num_transcripts


def num_genes(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    tables_key: str = "table",
    inplace: bool = True,
) -> int:
    """
    Counts the number of unique genes in the given SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing gene information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    int
        The number of unique genes found in the specified SpatialData object.
    """
    # converting from np.int64 to int for consistency
    num_genes = int(sdata.points[points_key][points_gene_key].nunique().compute())
    if inplace:
        sdata.tables[tables_key].uns["num_genes"] = num_genes
    return num_genes


def perc_unassigned_transcripts(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> float:
    """
    Calculates the proportion of unassigned transcripts in a SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The spatial data object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_cell_id_key : str, optional
        The key to access cell assignment information within the transcript data. Default is "cell_id".
    unassigned_key : int, optional
        The value indicating an unassigned transcript. Default is -1.
    points_background_id : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    float
        The fraction of transcripts that are unassigned.
    """
    counts = sdata.points[points_key][points_cell_id_key].compute().value_counts()
    num_unassigned = counts.get(points_background_id, 0)
    # converting from np.float64 to float for consistency
    perc_unassigned_transcripts = float(num_unassigned / counts.sum())
    if inplace:
        sdata.tables[tables_key].uns["perc_unassigned_transcripts"] = perc_unassigned_transcripts
    return perc_unassigned_transcripts


def transcripts_per_cell(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Counts the number of transcripts assigned to each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing transcript and cell assignment information.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    points_key : str, optional
        The key in `sdata.points` corresponding to transcript data. Default is "transcripts".
    points_cell_id_key : str, optional
        The column name in the transcript data that contains cell assignment information. Default is "cell_id".
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: the cell identifier (`cell_key`) and the
        corresponding transcript count ("transcript_count").
    """
    counts = sdata.points[points_key][points_cell_id_key].compute().value_counts().astype("int64")
    counts_df = counts.reset_index()
    counts_df.columns = [points_cell_id_key, "transcript_count"]

    if inplace:
        merge_into_obs(
            sdata, tables_key, counts_df, tables_cell_id_key, points_cell_id_key, fillna_cols=["transcript_count"]
        )

    return counts_df


def genes_per_cell(
    sdata,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the number of unique genes detected per cell.

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per cell, containing the cell identifier and
        the count of unique genes detected in that cell.
    """
    df = sdata.points[points_key].compute()
    # Group by cell and count unique genes
    gene_counts = df.groupby(points_cell_id_key)[points_gene_key].nunique().reset_index()
    gene_counts.columns = [points_cell_id_key, "gene_count"]
    if inplace:
        merge_into_obs(
            sdata, tables_key, gene_counts, tables_cell_id_key, points_cell_id_key, fillna_cols=["gene_count"]
        )

    return gene_counts


def transcript_density(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    points_key: str = "transcripts",
    tables_cell_id_key: str = "cell_id",
    tables_area_volume_key: str = "cell_area",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the transcript density for each cell in a SpatialData object.
    Transcript density is defined as the number of transcripts per unit area for each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    points_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    tables_cell_id_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".
    tables_area_volume_key: str, optional
        The key in the table indicating the cell area/volume. Default is "cell_area".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `[cell_key, "transcript_density"]`,
        where "transcript_density" is the number of transcripts per unit area for
        each cell. Rows with missing values are dropped.
    """
    adata = sdata.tables[tables_key]
    # this will also add the transcript counts inplace
    counts_df = transcripts_per_cell(sdata, points_key=points_key, tables_cell_id_key=tables_cell_id_key)
    area_df = adata.obs[[tables_cell_id_key, tables_area_volume_key]]

    merged = counts_df.merge(area_df, on=tables_cell_id_key, how="left")
    merged["transcript_density"] = merged["transcript_count"] / merged[tables_area_volume_key]

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            merged[[tables_cell_id_key, "transcript_density"]],
            tables_cell_id_key,
            tables_cell_id_key,
            fillna_cols=["transcript_density"],
        )

    return merged[[tables_cell_id_key, "transcript_density"]].dropna()


def morphological_features(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    shapes_key: str = "cell_boundaries",
    shapes_cell_id_key: str = "cell_id",
    features_to_compute: list | None = None,
    n_jobs: int = -1,  # number of parallel jobs, -1 uses all CPUs
    tables_key: str = "table",
    inplace: bool = True,
):
    """
    Compute morphological features for cell shapes in a spatial transcriptomics dataset.

    Parameters
    ----------
    sdata : object
        Spatial data object containing cell shape information. Must have a `.shapes` attribute with geometries.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with `shapes_cell_id_key`.
    shapes_key : str, optional
        Key in `sdata.shapes` specifying the geometry column (default is "cell_boundaries").
    shapes_cell_id_key : str, optional
        Key in `sdata.shapes` specifying the unique cell identifier column (default is "cell_id").
    features_to_compute : list of str, optional
        List of morphological features to compute. If None, all available features are computed.
        Available features: "cell_area", "perimeter", "circularity", "bbox_width", "bbox_height",
        "extent", "solidity", "convexity", "elongation", "eccentricity", "compactness", "num_polygons".
    n_jobs : int, optional
        Number of parallel jobs to use for computation. -1 uses all available CPUs (default is -1).
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    features : pandas.DataFrame
        DataFrame containing the computed morphological features for each cell, indexed by `shapes_cell_id_key`.

    Raises
    ------
    ValueError
        If any requested feature in `features_to_compute` is not recognized.

    Notes
    -----
    - Requires `geopandas`, `shapely`, `numpy`, `pandas`, and `joblib`.
    - Some features are proxies or approximations (e.g., "sphericity" uses "circularity").
    - Invalid or null geometries are filtered out before computation.
    """
    # Define all possible features
    all_features = [
        "cell_area",
        "perimeter",
        "circularity",
        "bbox_width",
        "bbox_height",
        "extent",
        "solidity",
        "convexity",
        "elongation",
        "eccentricity",
        "compactness",
        "num_polygons",
    ]

    # If no features specified, compute all
    if features_to_compute is None:
        features_to_compute = all_features
    else:
        # Validate features requested
        print(features_to_compute)
        invalid_feats = set(features_to_compute) - set(all_features)
        if invalid_feats:
            raise ValueError(f"Unknown features requested: {invalid_feats}")

    cells = sdata.shapes[shapes_key]
    if not isinstance(cells, gpd.GeoDataFrame):
        cells = cells.to_gdf()

    # Filter valid geometries
    cells = cells[cells.geometry.notnull() & cells.geometry.is_valid].copy()

    features = pd.DataFrame()
    if shapes_cell_id_key is None:
        features_cell_id = "cell_id"
        features[features_cell_id] = cells.index.values
    else:
        features_cell_id = shapes_cell_id_key
        features[features_cell_id] = cells[shapes_cell_id_key].values

    geom = cells.geometry

    # Compute features conditionally
    if "cell_area" in features_to_compute or any(
        f in features_to_compute for f in ["circularity", "extent", "solidity", "compactness", "sphericity"]
    ):
        areas = geom.area.values
        if "cell_area" in features_to_compute:
            features["cell_area"] = areas
    else:
        areas = None

    if "perimeter" in features_to_compute or any(
        f in features_to_compute
        for f in [
            "circularity",
            "compactness",
            "convexity",
            "compactness",
            "sphericity",
        ]
    ):
        perimeters = geom.length.values
        if "perimeter" in features_to_compute:
            features["perimeter"] = perimeters
    else:
        perimeters = None

    if "circularity" in features_to_compute:
        if areas is None:
            areas = geom.area.values
        if perimeters is None:
            perimeters = geom.length.values
        features["circularity"] = 4 * np.pi * areas / (perimeters**2 + 1e-6)

    if any(f in features_to_compute for f in ["bbox_width", "bbox_height", "extent"]):
        bounds = geom.bounds
        if "bbox_width" in features_to_compute:
            features["bbox_width"] = (bounds["maxx"] - bounds["minx"]).values
        if "bbox_height" in features_to_compute:
            features["bbox_height"] = (bounds["maxy"] - bounds["miny"]).values
        if "extent" in features_to_compute:
            width = (bounds["maxx"] - bounds["minx"]).values
            height = (bounds["maxy"] - bounds["miny"]).values
            if areas is None:
                areas = geom.area.values
            features["extent"] = areas / (width * height + 1e-6)

    if "solidity" in features_to_compute or "convexity" in features_to_compute:
        convex_hull = geom.convex_hull
        if "solidity" in features_to_compute:
            convex_areas = convex_hull.area.values
            if areas is None:
                areas = geom.area.values
            features["solidity"] = areas / (convex_areas + 1e-6)
        if "convexity" in features_to_compute:
            convex_perimeters = convex_hull.length
            if perimeters is None:
                perimeters = geom.length.values
            features["convexity"] = (convex_perimeters / (perimeters + 1e-6)).values

    # Parallelized elongation and eccentricity calculation
    def compute_elong_ecc(poly):
        if poly.is_empty:
            return np.nan, np.nan

        # Handle MultiPolygon by selecting the largest polygon by area
        if isinstance(poly, MultiPolygon):
            if len(poly.geoms) == 0:
                return np.nan, np.nan
            poly = max(poly.geoms, key=lambda p: p.area)

        # Skip invalid or degenerate geometries
        if not isinstance(poly, Polygon) or poly.area == 0:
            return np.nan, np.nan

        # Compute minimum rotated rectangle
        min_rect = poly.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)

        if len(coords) < 4:
            return np.nan, np.nan

        # Compute edge lengths
        edges = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])) for i in range(4)]
        edges = sorted(edges)
        if edges[1] == 0:
            return np.nan, np.nan

        # Elongation and eccentricity
        elongation = edges[2] / edges[1]
        a = edges[2] / 2
        b = edges[1] / 2
        eccentricity = np.sqrt(a**2 - b**2) / a if a > 0 else np.nan

        return elongation, eccentricity

    if "elongation" in features_to_compute or "eccentricity" in features_to_compute:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_elong_ecc)(poly) for poly in geom)
        elongations, eccentricities = zip(*results, strict=False)
        if "elongation" in features_to_compute:
            features["elongation"] = elongations
        if "eccentricity" in features_to_compute:
            features["eccentricity"] = eccentricities

    if "compactness" in features_to_compute:
        if perimeters is None:
            perimeters = geom.length.values
        if areas is None:
            areas = geom.area.values
        features["compactness"] = (perimeters**2) / (areas + 1e-6)

    if "num_polygons" in features_to_compute:
        features["num_polygons"] = geom.apply(count_polygons).values

    if inplace:
        merge_into_obs(sdata, tables_key, features, tables_cell_id_key, features_cell_id)

    return features
