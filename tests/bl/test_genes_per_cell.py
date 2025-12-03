import pandas as pd
import pytest

import segtraq as st


def test_genes_per_cell(sdata_new):
    genes_per_cell = st.bl.genes_per_cell(sdata_new)
    assert type(genes_per_cell) is pd.DataFrame, "Genes per cell should return a DataFrame"
    assert not genes_per_cell.empty, "Genes per cell DataFrame should not be empty"
    assert "cell_id" in genes_per_cell.columns, "DataFrame should contain 'cell_id' column"
    assert "gene_count" in genes_per_cell.columns, "DataFrame should contain 'gene_count' column"
    assert (
        "gene_count" in sdata_new.tables["table"].obs.columns
    ), "'gene_count' should be in the observation columns of the table"

    # TODO: reactivate once the test data is updated
    # num_cells = st.bl.num_cells(sdata_new)
    # assert genes_per_cell.shape[0] == num_cells, "Number of rows in DataFrame should match number of cells"


def test_genes_per_cell_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.genes_per_cell(sdata_new, points_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.genes_per_cell(sdata_new, points_cell_id_key="invalid_key")
