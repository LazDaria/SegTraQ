import pytest

import segtraq as st


def test_num_transcripts(sdata_new):
    num_transcripts = st.bl.num_transcripts(sdata_new)
    assert type(num_transcripts) is int, "Number of transcripts should be an integer"
    assert num_transcripts > 0, "Number of transcripts should be greater than zero"


def test_num_transcripts_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.num_transcripts(sdata_new, transcript_key="invalid_key")
