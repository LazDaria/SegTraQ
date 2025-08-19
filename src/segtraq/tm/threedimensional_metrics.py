import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.stats import kurtosis, skew

def shape_metrics(
    sdata: sd.SpatialData,
    transcript_key: str = "transcripts",
    z_coordinate: str = "z"
    ) -> pd.DataFrame:

    """
    Calculates shape statistics (mean, variance, skew, kurtosis) based on the first four moments of the distribution

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing spatial transcriptomics data.
    transcript_key : str, optional
        The key in the transcript table indicating transcript identifiers. Default is "transcripts".
    z_coordinate : str, optional
        The coordinate name of the z coordinate
    

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `["mean", "variance", "skew, "kurtosis"]`
    """
    
    
    z = np.array(sdata.points[transcript_key][z_coordinate])

    df_res = pd.DataFrame(mean = np.mean(z), variance = np.var(z), skew = skew(z), kurtosis = kurtosis(z))

    return(df_res)