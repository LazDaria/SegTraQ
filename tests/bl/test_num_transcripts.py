import pytest

import segtraq as st


def test_num_transcripts(sdata):
    num_transcripts = st.bl.num_transcripts(sdata)
    assert type(num_transcripts) is int, "Number of transcripts should be an integer"
    assert num_transcripts > 0, "Number of transcripts should be greater than zero"


def test_num_transcripts_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.num_transcripts(sdata, transcript_key="invalid_key")
