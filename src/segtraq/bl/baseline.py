import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from joblib import Parallel, delayed
from shapely.geometry import MultiPolygon, Polygon
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ..utils import _is_background, merge_into_obs, merge_into_var
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
    points = sdata.points[points_key]
    points = points.compute() if hasattr(points, "compute") else points
    num_transcripts = len(points)
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
    points = sdata.points[points_key]
    points = points.compute() if hasattr(points, "compute") else points
    num_genes = int(points[points_gene_key].nunique())
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
    Calculates the percentage of unassigned transcripts in a SpatialData object.

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
    points = sdata.points[points_key][points_cell_id_key]
    is_background = _is_background(points, points_background_id)
    is_background = is_background.compute() if hasattr(is_background, "compute") else is_background
    perc_unassigned_transcripts = is_background.mean() * 100

    if inplace:
        sdata.tables[tables_key].uns["perc_unassigned_transcripts"] = perc_unassigned_transcripts

    # converting from np.float to float
    return float(perc_unassigned_transcripts)


def perc_unassigned_transcripts_per_gene(
    sdata: sd.SpatialData,
    points_key: str = "transcripts",
    points_gene_key: str = "feature_name",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the number and percentage of unassigned transcripts per gene in a SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The spatial data object containing transcript information.
    points_key : str, optional
        The key to access transcript data within the spatial data object. Default is "transcripts".
    points_gene_key : str, optional
        The key for gene names in the transcript data. Default is "feature_name".
    points_cell_id_key : str, optional
        The key for cell assignment information within the transcript data. Default is "cell_id".
    points_background_id : int, optional
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, stores the resulting DataFrame in
        `sdata.tables[tables_key].uns["perc_unassigned_transcripts_per_gene"]`.
        Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by gene name with columns:
        - 'total' : total number of transcripts for the gene
        - 'unassigned' : number of unassigned transcripts
        - 'perc_unassigned' : percentage of unassigned transcripts
    """
    points = sdata.points[points_key]

    # Compute only necessary columns
    df = points[[points_gene_key, points_cell_id_key]].compute()

    # Aggregate total and unassigned counts efficiently
    result = (
        df.groupby(points_gene_key, observed=True)[points_cell_id_key]
        .agg(
            total="count",
            unassigned=lambda x: _is_background(x, points_background_id).sum(),
        )
        .astype(int)
    )

    # Compute percentage
    result["perc_unassigned"] = result["unassigned"] / result["total"] * 100

    # Store and return
    if inplace:
        merge_into_var(sdata, tables_key, result)

    return result


def transcripts_per_cell(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Counts the number of transcripts assigned to each cell (excluding unassigned transcripts).

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing transcript and cell assignment information.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key in `sdata.points` corresponding to transcript data. Default is "transcripts".
    points_cell_id_key : str, optional
        The column name in the transcript data that contains cell assignment information. Default is "cell_id".
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
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
    counts = sdata.points[points_key][points_cell_id_key]
    counts = counts.compute() if hasattr(counts, "compute") else counts
    counts = counts[~_is_background(counts, points_background_id)].value_counts().astype("int64")
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
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the number of unique genes detected per cell (excluding unassigned transcripts).

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
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
    # Exclude unassigned transcripts
    df = df[~_is_background(df[points_cell_id_key], points_background_id)]

    # Group by cell and count unique genes
    gene_counts = df.groupby(points_cell_id_key)[points_gene_key].nunique().reset_index()
    gene_counts.columns = [points_cell_id_key, "gene_count"]
    if inplace:
        merge_into_obs(
            sdata, tables_key, gene_counts, tables_cell_id_key, points_cell_id_key, fillna_cols=["gene_count"]
        )

    return gene_counts


def mean_transcripts_per_gene_per_cell(
    sdata: sd.SpatialData,
    tables_cell_id_key: str = "cell_id",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_gene_key: str = "feature_name",
    points_background_id: int = -1,
    tables_key: str = "table",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Computes the mean number of transcripts per gene per cell (excluding unassigned transcripts).

    Transcripts are first counted per (cell, gene). Then, for each cell,
    we compute the mean of these per-gene transcript counts across genes
    detected in that cell.

    Notes
    -----
    This mean is computed across *detected* genes only (i.e., genes with at least
    one transcript in the cell). Genes with zero transcripts in a cell are not included.

    Parameters
    ----------
    sdata : object
        An object containing spatial transcriptomics data with a `points` attribute.
    tables_cell_id_key : str
        Column in `sdata.tables[tables_key].obs containing cell IDs to match with sdata.shapes[shapes_key] index.
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_gene_key : str, optional
        The column name in the transcript data representing gene names (default is "feature_name").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per cell containing the mean transcripts per detected gene:
        columns are `[points_cell_id_key, "mean_transcripts_per_gene"]`.
    """
    df = sdata.points[points_key].compute()
    # Exclude unassigned transcripts
    df = df[~_is_background(df[points_cell_id_key], points_background_id)]

    # Count transcripts per (cell, gene)
    per_gene_counts = (
        df.groupby([points_cell_id_key, points_gene_key], observed=True).size().reset_index(name="transcript_count")
    )

    # For each cell, compute mean transcripts across detected genes
    per_cell_mean = (
        per_gene_counts.groupby(points_cell_id_key, observed=True)["transcript_count"]
        .mean()
        .reset_index(name="mean_transcripts_per_gene")
    )

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            per_cell_mean,
            tables_cell_id_key,
            points_cell_id_key,
            fillna_cols=["mean_transcripts_per_gene"],
        )

    return per_cell_mean


def transcript_density(
    sdata: sd.SpatialData,
    tables_key: str = "table",
    tables_cell_id_key: str = "cell_id",
    tables_area_key: str = "cell_area",
    points_key: str = "transcripts",
    points_cell_id_key: str = "cell_id",
    points_background_id: int = -1,
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
    tables_cell_id_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".
    tables_area_key: str, optional
        The key in the table indicating the cell area. Default is "cell_area".
    points_key : str, optional
        The key to access the transcript data within `sdata.points` (default is "transcripts").
    points_cell_id_key : str, optional
        The column name in the transcript data representing cell identifiers (default is "cell_id").
    points_background_id: int = -1,
        The value indicating an unassigned transcript. Default is -1.
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
    counts_df = transcripts_per_cell(
        sdata,
        points_key=points_key,
        tables_cell_id_key=tables_cell_id_key,
        points_background_id=points_background_id,
        points_cell_id_key=points_cell_id_key,
        tables_key=tables_key,
    )
    area_df = adata.obs[[tables_cell_id_key, tables_area_key]]

    merged = counts_df.merge(area_df, left_on=points_cell_id_key, right_on=tables_cell_id_key, how="left")
    merged["transcript_density"] = merged["transcript_count"] / merged[tables_area_key]

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
    tables_centroid_x_key: str = "centroid_x",
    tables_centroid_y_key: str = "centroid_y",
    shapes_key: str = "cell_boundaries",
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
    tables_centroid_x_key : str, optional
        Column in `sdata.tables[tables_key].obs` to store the x-coordinate of the centroid (default is "centroid_x").
    tables_centroid_y_key : str, optional
        Column in `sdata.tables[tables_key].obs` to store the y-coordinate of the centroid (default is "centroid_y").
    shapes_key : str, optional
        Key in `sdata.shapes` specifying the geometry column (default is "cell_boundaries").
    features_to_compute : list of str, optional
        List of morphological features to compute. If None, all available features are computed.
        Available features: "centroid", "cell_area", "perimeter", "circularity", "bbox_width", "bbox_height",
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
        DataFrame containing the computed morphological features for each cell, indexed by sdata[shapes_key].index.name.

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
        "centroid",
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
    # if we already have centroids in the table, we do not compute them here
    if features_to_compute is None and (tables_centroid_x_key is not None and tables_centroid_y_key is not None):
        all_features.remove("centroid")

    # If no features specified, compute all
    if features_to_compute is None:
        features_to_compute = all_features
    else:
        # Validate features requested
        invalid_feats = set(features_to_compute) - set(all_features)
        if invalid_feats:
            raise ValueError(f"Unknown features requested: {invalid_feats}")

    cells = sdata.shapes[shapes_key]
    if not isinstance(cells, gpd.GeoDataFrame):
        cells = cells.to_gdf()

    # Filter valid geometries
    cells = cells[cells.geometry.notnull() & cells.geometry.is_valid].copy()

    features = pd.DataFrame()

    features[cells.index.name] = cells.index.values

    geom = cells.geometry

    if "centroid" in features_to_compute:
        centroids = geom.centroid
        features["centroid_x"] = centroids.x.values
        features["centroid_y"] = centroids.y.values

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
        merge_into_obs(sdata, tables_key, features, tables_cell_id_key, cells.index.name)

    return features

def image_features(
    sdata: sd.SpatialData,
    images_key: str,
    shapes_key: str,
    channel_names: str | list[str] | None = None,
    features: list[str] = ("mean", "std", "median", "min", "max"),
    shapes_cell_id_key: str = "cell_id",
    tables_key: str | None = "table",
    tables_cell_id_key: str = "cell_id",
    inplace : bool = True
) -> pd.DataFrame:
    """
    Compute image-based features for each cell polygon in a SpatialData object.

    For each cell in `sdata.shapes[shapes_key]`, the method masks the cell's
    region in `sdata.images[images_key]` and computes per-channel summary
    statistics over the masked pixels.
    
    Note that this requires rasterizing the cell polygons.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object containing images and shapes.
    images_key : str
        Key in `sdata.images` for the image to sample from (e.g. "image").
        Expected to be a DataArray with dimensions (c, y, x).
    shapes_key : str
        Key in `sdata.shapes` for the cell polygons (e.g. "cell_boundaries").
    channel_names : list[str] | None
        Names for the image channels. If None, channels are named c0, c1, ...
        Must match the number of channels in the image if provided.
    features : list[str]
        Which statistics to compute per channel per cell.
        Supported: "mean", "std", "median", "min", "max"
    shapes_cell_id_key : str
        Column in `sdata.shapes[shapes_key]` that holds the cell ID.
        If the column does not exist, the GeoDataFrame index is used instead.
    tables_key : str | None
        If provided, the returned DataFrame is also stored in
        `sdata.tables[tables_key].obsm["image_features"]` keyed by cell_id,
        aligned to the table's obs index. Set to None to skip.
    inplace : bool, optional
        If True, modifies the SpatialData object in place. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by cell id with columns
        ``{channel}_{feature}`` for every requested feature and channel.

    Examples
    --------
    >>> features_df = image_features(
    ...     sdata,
    ...     images_key="image",
    ...     shapes_key="cell_boundaries",
    ...     channel_names=["DAPI"],
    ...     features=["mean", "std"],
    ... )
    >>> features_df["DAPI_mean"].hist()
    """
    supported_features = {"mean", "std", "median", "min", "max"}
    bad_features = set(features) - supported_features
    if bad_features:
        raise ValueError(f"Unsupported features: {bad_features}. Please choose from {supported_features}.")

    # ------------------------------------------------------------------ #
    # Load image as a numpy array (c, y, x)                              #
    # ------------------------------------------------------------------ #
    if images_key is None:
        raise ValueError("images_key must be provided to compute image features. Please do so when initializing the SegTraQ object.")
    if images_key not in sdata.images:
        raise KeyError(f"Image key '{images_key}' not found in sdata.images. Available keys: {list(sdata.images.keys())}")
    image_da = sdata.images[images_key]
    # converting the data array into a numpy array
    image_np = image_da.compute().values  # (C, H, W)
    n_channels, img_h, img_w = image_np.shape

    if channel_names is None:
        channel_names = [f"c{i}" for i in range(n_channels)]
    if isinstance(channel_names, str):
        channel_names = [channel_names]
    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries ({channel_names}) but image has {n_channels} channels. "
            f"Please adjust the channel_names argument accordingly."
        )

    # ------------------------------------------------------------------ #
    # Retrieve shapes and resolve cell ids                               #
    # ------------------------------------------------------------------ #
    shapes = sdata.shapes[shapes_key].copy()

    if shapes_cell_id_key in shapes.columns:
        cell_ids = shapes[shapes_cell_id_key].values
    else:
        cell_ids = shapes.index.values

    # ------------------------------------------------------------------ #
    # Build coordinate → pixel mapping from the image's transform        #
    # xarray stores spatial coordinates in the "x" and "y" dimensions    #
    # ------------------------------------------------------------------ #
    x_coords = image_da.coords["x"].values  # length W
    y_coords = image_da.coords["y"].values  # length H

    x_min, x_max = float(x_coords[0]), float(x_coords[-1])
    y_min, y_max = float(y_coords[0]), float(y_coords[-1])

    # pixel size (handle descending y axis)
    px_w = (x_max - x_min) / (img_w - 1)  # coords per pixel, x
    px_h = (y_max - y_min) / (img_h - 1)  # coords per pixel, y (may be negative)

    # ------------------------------------------------------------------ #
    # Iterate over cells and compute features                            #
    # ------------------------------------------------------------------ #
    _FEAT_FUNCS = {
        "mean":   np.mean,
        "std":    np.std,
        "median": np.median,
        "min":    np.min,
        "max":    np.max,
    }

    records = []
    for cell_id, geom in zip(cell_ids, shapes.geometry):
        row: dict = {shapes_cell_id_key: cell_id}

        if geom is None or geom.is_empty:
            for ch in channel_names:
                for feat in features:
                    row[f"{ch}_{feat}"] = np.nan
            records.append(row)
            continue
            
        # Bounding box of this cell in spatial coordinates
        minx, miny, maxx, maxy = geom.bounds

        # Convert bbox to pixel coordinates
        col_min = int(np.clip(np.floor((minx - x_min) / px_w), 0, img_w - 1))
        col_max = int(np.clip(np.ceil((maxx - x_min) / px_w), 0, img_w - 1))
        row_min = int(np.clip(np.floor((miny - y_min) / (px_h if px_h != 0 else 1)), 0, img_h - 1))
        row_max = int(np.clip(np.ceil((maxy - y_min) / (px_h if px_h != 0 else 1)), 0, img_h - 1))

        # Ensure valid, non-empty patch
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min
        if row_min == row_max:
            row_max = min(row_min + 1, img_h - 1)
        if col_min == col_max:
            col_max = min(col_min + 1, img_w - 1)

        patch_h = row_max - row_min + 1
        patch_w = col_max - col_min + 1

        # Rasterize the polygon into a boolean mask over the patch
        patch_x_min = x_min + col_min * px_w
        patch_x_max = x_min + col_max * px_w
        patch_y_min = y_min + row_min * (px_h if px_h != 0 else 1)
        patch_y_max = y_min + row_max * (px_h if px_h != 0 else 1)

        affine = from_bounds(
            patch_x_min - px_w / 2,
            min(patch_y_min, patch_y_max) - abs(px_h) / 2,
            patch_x_max + px_w / 2,
            max(patch_y_min, patch_y_max) + abs(px_h) / 2,
            patch_w,
            patch_h,
        )

        mask = rasterize(
            [(geom, 1)],
            out_shape=(patch_h, patch_w),
            transform=affine,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        # Sample each channel and compute statistics
        for ch_idx, ch_name in enumerate(channel_names):
            patch = image_np[ch_idx, row_min:row_max + 1, col_min:col_max + 1]
            pixel_vals = patch[mask].astype(np.float64)

            # guard against empty masks
            if pixel_vals.size == 0:
                for feat in features:
                    row[f"{ch_name}_{feat}"] = np.nan
                continue

            for feat in features:
                row[f"{ch_name}_{feat}"] = float(_FEAT_FUNCS[feat](pixel_vals))

        records.append(row)

    result_df = pd.DataFrame(records).set_index(shapes_cell_id_key)

    # ------------------------------------------------------------------ #
    # Optionally attach to the AnnData table                             #
    # ------------------------------------------------------------------ #
    if inplace:
        merge_into_obs(
            sdata, tables_key, result_df, tables_cell_id_key, shapes_cell_id_key
        )

    return result_df
