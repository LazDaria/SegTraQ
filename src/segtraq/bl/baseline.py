import spatialdata as sd


def num_cells(sdata: sd.SpatialData, shape_key: str = "cell_boundaries"):
    return len(sdata.shapes[shape_key])


def transcripts_per_cell(sdata: sd.SpatialData, transcript_key: str = "transcripts", cell_key: str = "cell_id"):
    transcript_counts = sdata.points[transcript_key][cell_key].compute().value_counts().reset_index()
    transcript_counts.columns = ["cell_id", "transcript_count"]
    return transcript_counts
