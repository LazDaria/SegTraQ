import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import LinearRing, Point

from ..utils import merge_into_obs


def centroid_mean_coord_diff(
    sdata: sd.SpatialData,
    feature,
    tables_key: str = "table",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    tables_cell_id_key: str = "cell_id",
    points_cell_id_key: str = "cell_id",
    shapes_cell_id_key: str = "cell_id",
    points_x_key: str = "x",
    points_y_key: str = "y",
    shapes_key: str = "cell_boundaries",
    centroid_key: list = ("centroid_x", "centroid_y"),
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Calculates the euclidean distance between the mean x,y coordinate of the transcripts
    indicated by the feature variable and the centroid of each cell.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    feature: str
        String indicating the feature/gene to calculate the mean transcript coordiantes on
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    shapes_cell_id_key : str or None, optional, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    shapes_key: str, optional
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    centroid_key: list, optional
        The keys to access the centroids in the `sdata.shapes` slot. Defaults are "centroid_x" and "centroid_y"
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["centroid_x, "centroid_y", "x", "y", "distance"]`,
        where "distance" is the euclidean distance between the coordinates `["centroid_x, "centroid_y"] and
        ["x", "y"].

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.
    """

    # extract the transcript information
    df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[points_cell_id_key].isin(sdata[tables_key].obs[points_cell_id_key])]

    # subset transcript dataframe to the feature
    df = df[df[points_gene_key] == feature]

    # drop the background transcripts in cell_id == -1
    df = df[df[points_cell_id_key] != -1]

    # group by cell id
    df = df.groupby(points_cell_id_key)

    # compute the mean x,y coordiantes of the transcripts per cell
    x_mean = df[points_x_key].mean()
    y_mean = df[points_y_key].mean()

    x_mean = pd.DataFrame(x_mean)
    y_mean = pd.DataFrame(y_mean)

    gdf = sdata[shapes_key].copy()

    if shapes_cell_id_key is not None:
        shapes_cell_id_key_fixed = shapes_cell_id_key
        gdf.set_index(shapes_cell_id_key_fixed, drop=True, inplace=True)
    else:
        shapes_cell_id_key_fixed = "cell_id"
        gdf.index.name = shapes_cell_id_key_fixed

    # extract the centroids
    df_centroids_x = pd.DataFrame(gdf.centroid.x, columns=[centroid_key[0]])
    df_centroids_y = pd.DataFrame(gdf.centroid.y, columns=[centroid_key[1]])

    # do an inner merge on the cell ids - some cells have no transcripts
    df_total_x = df_centroids_x.merge(
        x_mean, left_on=shapes_cell_id_key_fixed, right_on=points_cell_id_key, how="inner"
    )
    df_total_y = df_centroids_y.merge(
        y_mean, left_on=shapes_cell_id_key_fixed, right_on=points_cell_id_key, how="inner"
    )

    df_total = pd.concat([df_total_x, df_total_y], axis=1)

    # calculate the euclidean distance
    df_total["distance"] = np.linalg.norm(
        df_total.loc[:, [centroid_key[0], centroid_key[1]]].values
        - df_total.loc[:, [points_y_key, points_x_key]].values,
        ord=2,
        axis=1,
    )

    # extract the cell area
    area_df = sdata[tables_key].obs[[tables_cell_id_key, "cell_area"]]
    df_total = df_total.merge(area_df, left_on=points_cell_id_key, right_on=tables_cell_id_key, how="left")

    # normalise the cell area
    df_total[f"distance_{feature}"] = df_total["distance"] / df_total["cell_area"]
    df_total = df_total.reset_index(drop=True)

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            df_total[[points_cell_id_key, f"distance_{feature}"]],
            tables_cell_id_key,
            points_cell_id_key,
        )

    return df_total


def distance_to_membrane(
    sdata: sd.SpatialData,
    feature,
    tables_key: str = "table",
    points_gene_key: str = "feature_name",
    points_key: str = "transcripts",
    points_x_key: str = "x",
    points_y_key: str = "y",
    points_cell_id_key: str = "cell_id",
    tables_cell_id_key: str = "cell_id",
    shapes_cell_id_key: str = "cell_id",
    inplace: bool = True,
):
    """
    Calculates the mean distance of the transcript of a feature of interest to the outline of the cell segmentation

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    feature: str
        String indicating the feature/gene to calculate the mean transcript coordiantes on
    tables_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    points_gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    points_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    points_x_key : str, default="x"
        Column for the x-coordinate of each transcript/spot.
    points_y_key : str, default="y"
        Column for the y-coordinate of each transcript/spot.
    tables_cell_id_key : str, default="cell_id"
        Column in the cell table uniquely identifying each cell.
    points_cell_id_key : str, default="cell_id"
        Column in the points table linking each transcript/spot to a cell.
    shapes_cell_id_key : str or None, optional, default="cell_id"
        Column in the cell-boundary shapes linking polygons to cell IDs.
        If `None`, the shape index is used as the cell ID.
    inplace : bool, optional
        Whether to add the results to `sdata.tables`. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["distance_to_outline_inverse", f"distance_to_outline_{feature}" and "cell_area"]`

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.

    """

    # extract the transcript information
    df = sdata.points[points_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[points_cell_id_key].isin(sdata[tables_key].obs[tables_cell_id_key])]

    # subset transcript dataframe to the feature
    df = df[df[points_gene_key] == feature]

    # drop the background transcripts in cell_id == -1
    df = df[df[points_cell_id_key] != -1]

    # zip the coordinates to a common column as tuple
    df["coordinates"] = list(zip(df[points_x_key], df[points_y_key], strict=False))

    # make the coordinates into a Point object
    df["coordinate_points"] = df["coordinates"].map(lambda x: Point(x))

    # extract the cell segmentation boundaries
    gdf = sdata["cell_boundaries"]

    # make the cell key the index for merging the two dataframes
    # df = df.set_index(df[cell_key])

    # merge the geopandas dataframe with the dataframe from above

    if shapes_cell_id_key is not None:
        shapes_cell_id_key_fixed = shapes_cell_id_key
        gdf.set_index(shapes_cell_id_key_fixed, drop=True, inplace=True)
    else:
        shapes_cell_id_key_fixed = "cell_id"
        gdf.index.name = shapes_cell_id_key_fixed

    gdf = gdf.merge(df, how="inner", left_on=points_cell_id_key, right_on=shapes_cell_id_key_fixed)

    # compute the linear outline of the cell segmentation
    gdf["linear_geometry"] = gdf.apply(lambda x: LinearRing(x["geometry"].exterior.coords), axis=1)

    # drop NaN values in the coordinate point column
    gdf = gdf.dropna(subset="coordinate_points")

    # calculate the distance of the transcript points to the linear segment
    gdf[f"distance_to_outline_{feature}"] = gdf.apply(
        lambda x: x["coordinate_points"].distance(x["linear_geometry"]), axis=1
    )

    # calculate the mean transcript distance to the cell outline per cell
    mean_distance_to_outline = gdf.groupby(shapes_cell_id_key)[[f"distance_to_outline_{feature}"]].mean()

    # extract the cell area
    area_df = sdata[tables_key].obs[[tables_cell_id_key, "cell_area"]]
    mean_distance_to_outline = mean_distance_to_outline.merge(
        area_df, left_on=shapes_cell_id_key, right_on=tables_cell_id_key, how="left"
    )

    # normalise by area
    mean_distance_to_outline[f"distance_to_outline_inverse_{feature}"] = (
        mean_distance_to_outline[f"distance_to_outline_{feature}"] / mean_distance_to_outline["cell_area"]
    )

    # take the inverse - score is high when distance is small. sqrt transformed to handle right skewed distribution
    mean_distance_to_outline[f"distance_to_outline_inverse_{feature}"] = 1 / np.sqrt(
        mean_distance_to_outline[f"distance_to_outline_{feature}"]
    )

    if inplace:
        merge_into_obs(
            sdata,
            tables_key,
            mean_distance_to_outline[
                [shapes_cell_id_key, f"distance_to_outline_{feature}", f"distance_to_outline_inverse_{feature}"]
            ],
            tables_cell_id_key,
            shapes_cell_id_key,
        )

    return mean_distance_to_outline
