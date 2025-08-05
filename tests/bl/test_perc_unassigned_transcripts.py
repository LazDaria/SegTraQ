import pytest

import segtraq as st


def test_perc_unassigned_transcripts(sdata):
    perc_unassigned = st.bl.perc_unassigned_transcripts(sdata)
    assert type(perc_unassigned) is float, "Percentage of unassigned transcripts should be a float"
    assert 0 <= perc_unassigned <= 1, "Percentage of unassigned transcripts should be between 0 and 1"


def test_perc_unassigned_transcripts_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata, transcript_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata, cell_key="invalid_key")
