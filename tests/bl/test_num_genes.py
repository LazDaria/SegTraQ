import pytest

import segtraq as st


def test_num_genes(sdata_new):
    num_genes = st.bl.num_genes(sdata_new)
    assert type(num_genes) is int, "Number of genes should be an integer"
    assert num_genes > 0, "Number of genes should be greater than zero"
    assert "num_genes" in sdata_new.tables["table"].uns.keys(), "'num_genes' should be present in uns"


def test_num_genes_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.num_genes(sdata_new, points_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.num_genes(sdata_new, points_gene_key="invalid_key")
