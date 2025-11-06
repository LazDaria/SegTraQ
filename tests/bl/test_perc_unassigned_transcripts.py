import pytest

import segtraq as st


def test_perc_unassigned_transcripts(sdata_new):
    perc_unassigned = st.bl.perc_unassigned_transcripts(sdata_new)
    assert type(perc_unassigned) is float, "Percentage of unassigned transcripts should be a float"
    assert 0 <= perc_unassigned <= 1, "Percentage of unassigned transcripts should be between 0 and 1"
    assert "perc_unassigned_transcripts" in sdata_new.tables["table"].uns.keys(), (
        "'perc_unassigned_transcripts' should be present in uns"
    )


def test_perc_unassigned_transcripts_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata_new, points_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts(sdata_new, points_cell_id_key="invalid_key")
