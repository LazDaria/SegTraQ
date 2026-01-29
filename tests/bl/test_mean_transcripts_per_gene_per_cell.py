import numpy as np

import segtraq as st


def test_mean_transcripts_per_gene_per_cell(sdata_new):
    mean_tx_per_gene_per_cell = st.bl.mean_transcripts_per_gene_per_cell(sdata_new)
    assert np.all(mean_tx_per_gene_per_cell["mean_transcripts_per_gene"] > 0), (
        "Mean transcripts per gene per cell should be greater than zero"
    )
