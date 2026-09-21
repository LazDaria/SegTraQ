import numpy as np
from ..utils import _get_genes

def _get_segtraq_markers(
    adata,
    markers: dict[str, dict[str, list[str]]] | None,
    tables_gene_key: str | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Return provided markers or load stored SegTraQ markers from ``adata.uns``."""
    if markers is not None:
        return markers

    if "segtraq_markers" not in adata.uns:
        raise ValueError(
            "No markers were provided and no stored SegTraQ markers were found in "
            "adata.uns['segtraq_markers']. Run markers_from_reference(..., inplace=True) "
            "first or pass markers explicitly."
        )

    genes = _get_genes(
        adata=adata,
        gene_key=tables_gene_key,
    )

    stored = adata.uns["segtraq_markers"]

    return {
        cell_type: {
            "positive": genes[np.asarray(marker_sets["positive"], dtype=int)].tolist(),
            "negative": genes[np.asarray(marker_sets["negative"], dtype=int)].tolist(),
        }
        for cell_type, marker_sets in stored.items()
    }