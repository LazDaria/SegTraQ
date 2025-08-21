import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import LinearRing, Point


def centroid_mean_coord_diff(
    sdata: sd.SpatialData,
    feature,
    table_key: str = "table",
    gene_key: str = "feature_name",
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
    coordinate_key: list = ("x", "y"),
    shape_key: str = "cell_boundaries",
    centroid_key: list = ("centroid_x", "centroid_y"),
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
    table_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    transcript_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    cell_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".
    coordinate_key: list, optional
        The keys to access the coordinates from the `sdata` `transcript_key` table. Defaults are "x" and "y"
    shape_key: str, optional
        The key in `sdata.shapes` specifying the geometry column. Default is "cell_boundaries".
    centroid_key: list, optional
        The keys to access the centroids in the `sdata.shapes` slot. Defaults are "centroid_x" and "centroid_y"

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
    df = sdata.points[transcript_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[cell_key].isin(sdata[table_key].obs[cell_key])]

    # subset transcript dataframe to the feature
    df = df[df[gene_key] == feature]

    # drop the background transcripts in cell_id == -1
    df = df[df[cell_key] != -1]

    # group by cell id
    df = df.groupby(cell_key)

    # compute the mean x,y coordiantes of the transcripts per cell
    x_mean = df[coordinate_key[0]].mean()
    y_mean = df[coordinate_key[1]].mean()

    x_mean = pd.DataFrame(x_mean)
    y_mean = pd.DataFrame(y_mean)

    # extract the centroids
    df_centroids_x = pd.DataFrame(sdata[shape_key].centroid.x, columns=[centroid_key[0]])
    df_centroids_y = pd.DataFrame(sdata[shape_key].centroid.y, columns=[centroid_key[1]])

    # do an inner join on the cell ids - some cells have no transcripts
    df_total_x = df_centroids_x.join(x_mean, on=cell_key, how="inner")
    df_total_y = df_centroids_y.join(y_mean, on=cell_key, how="inner")

    df_total = pd.concat([df_total_x, df_total_y], axis=1)

    # calculate the euclidean distance
    df_total["distance"] = np.linalg.norm(
        df_total.loc[:, [centroid_key[0], centroid_key[1]]].values
        - df_total.loc[:, [coordinate_key[0], coordinate_key[1]]].values,
        axis=1,
    )

    # extract the cell area
    area_df = sdata[table_key].obs[[cell_key, "cell_area"]]
    df_total = df_total.merge(area_df, on=cell_key, how="left")

    # normalise the cell area
    df_total["distance"] = df_total["distance"] / df_total["cell_area"]
    df_total = df_total.set_index(df_total[cell_key])
    return df_total


def distance_to_membrane(
    sdata: sd.SpatialData,
    feature,
    table_key: str = "table",
    gene_key: str = "feature_name",
    transcript_key: str = "transcripts",
    cell_key: str = "cell_id",
):
    """
    Calculates the mean distance of the transcript of a feature of interest to the outline of the cell segmentation

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    feature: str
        String indicating the feature/gene to calculate the mean transcript coordiantes on
    table_key : str, optional
        The key to access the AnnData table from `sdata.tables`. Default is "table".
    gene_key : str, optional
        The key to access gene names within the transcript data. Default is "feature_name".
    transcript_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    cell_key : str, optional
        The key in the table indicating cell identifiers. Default is "cell_id".

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["distance_to_outline_inverse", "distance_to_outline" and "cell_area"]`

    Notes
    -----
    Requires that the input AnnData table contains a "cell_area" column in `.obs`.

    """

    # extract the transcript information
    df = sdata.points[transcript_key].compute()

    # filter to those cells which are in the anndata object
    df = df[df[cell_key].isin(sdata[table_key].obs[cell_key])]

    # subset transcript dataframe to the feature
    df = df[df[gene_key] == feature]

    # drop the background transcripts in cell_id == -1
    df = df[df[cell_key] != -1]

    # zip the coordinates to a common column as tuple
    df["coordinates"] = list(zip(df["x"], df["y"], strict=False))

    # make the coordinates into a Point object
    df["coordinate_points"] = df["coordinates"].map(lambda x: Point(x))

    # extract the cell segmentation boundaries
    gdf = sdata["cell_boundaries"]

    # make the cell key the index for joining the two dataframes
    df = df.set_index(df[cell_key])

    # merge the geopandas dataframe with the dataframe from above
    gdf = gdf.join(df)

    # compute the linear outline of the cell segmentation
    gdf["linear_geometry"] = gdf.apply(lambda x: LinearRing(x["geometry"].exterior.coords), axis=1)

    # drop NaN values in the coordinate point column
    gdf = gdf.dropna(subset="coordinate_points")

    # calculate the distance of the transcript points to the linear segment
    gdf["distance_to_outline"] = gdf.apply(lambda x: x["coordinate_points"].distance(x["linear_geometry"]), axis=1)

    # rename index as this should not have the same name as one of the columns
    gdf.index.name = "index"

    # calculate the mean transcript distance to the cell outline per cell
    mean_distance_to_outline = gdf.groupby(cell_key)[["distance_to_outline"]].mean()

    # extract the cell area
    area_df = sdata[table_key].obs[[cell_key, "cell_area"]]
    mean_distance_to_outline = mean_distance_to_outline.merge(area_df, on=cell_key, how="left")

    # normalise by area
    mean_distance_to_outline["distance_to_outline_inverse"] = (
        mean_distance_to_outline["distance_to_outline"] / mean_distance_to_outline["cell_area"]
    )

    # take the inverse - score is high when distance is small. sqrt transformed to handle right skewed distribution
    mean_distance_to_outline["distance_to_outline_inverse"] = 1 / np.sqrt(
        mean_distance_to_outline["distance_to_outline"]
    )
    mean_distance_to_outline = mean_distance_to_outline.set_index(mean_distance_to_outline[cell_key])

    return mean_distance_to_outline
