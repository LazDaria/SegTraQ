import pandas as pd
import pytest

import segtraq as st


def test_transcripts_per_cell(sdata_new):
    transcripts_per_cell = st.bl.transcripts_per_cell(sdata_new)
    assert type(transcripts_per_cell) is pd.DataFrame, "Transcripts per cell should return a DataFrame"
    assert not transcripts_per_cell.empty, "Transcripts per cell DataFrame should not be empty"
    assert "cell_id" in transcripts_per_cell.columns, "DataFrame should contain 'cell_id' column"
    assert "transcript_count" in transcripts_per_cell.columns, "DataFrame should contain 'transcript_count' column"

    # TODO: reactivate once the test data is updated
    # num_cells = st.bl.num_cells(sdata_new)
    # The -1 is to account for transcripts not assigned to a cell
    # assert transcripts_per_cell.shape[0] - 1 == num_cells, "Number of rows in DataFrame should match number of cells"


def test_transcripts_per_cell_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.transcripts_per_cell(sdata_new, transcript_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.transcripts_per_cell(sdata_new, cell_key="invalid_key")
