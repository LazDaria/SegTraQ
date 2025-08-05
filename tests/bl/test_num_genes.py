import pytest

import segtraq as st


def test_num_genes(sdata):
    num_genes = st.bl.num_genes(sdata)
    assert type(num_genes) is int, "Number of genes should be an integer"
    assert num_genes > 0, "Number of genes should be greater than zero"


def test_num_genes_invalid_key(sdata):
    with pytest.raises(KeyError):
        st.bl.num_genes(sdata, transcript_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.num_genes(sdata, gene_key="invalid_key")
