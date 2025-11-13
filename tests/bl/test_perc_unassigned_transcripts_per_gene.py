import numpy as np
import pytest

import segtraq as st


def test_perc_unassigned_transcripts_per_gene(sdata_new):
    perc_unassigned = st.bl.perc_unassigned_transcripts_per_gene(sdata_new)
    arr = perc_unassigned["perc_unassigned"].values
    assert np.all((arr >= 0) & (arr <= 100)), "Percentage of unassigned transcripts should be between 0 and 100"
    assert "perc_unassigned" in sdata_new.tables["table"].var.columns, "'perc_unassigned' should be present in var"


def test_perc_unassigned_transcripts_per_gene_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_cell_id_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_gene_key="invalid_key")
