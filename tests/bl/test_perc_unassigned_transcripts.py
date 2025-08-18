import pytest

import segtraq as st


def test_perc_unassigned_transcripts(sdata_new):
    perc_unassigned = st.bl.perc_unassigned_transcripts(sdata_new)
    assert type(perc_unassigned) is float, "Percentage of unassigned transcripts should be a float"
    assert 0 <= perc_unassigned <= 1, "Percentage of unassigned transcripts should be between 0 and 1"


def test_perc_unassigned_transcripts_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata_new, transcript_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata_new, cell_key="invalid_key")
