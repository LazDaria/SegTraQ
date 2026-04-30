from ..utils import _looks_like_counts


def _get_count_matrix(adata, layer=None, tables_key="table"):
    if layer is None:
        X = adata.X
        if _looks_like_counts(X):
            return X
        if "counts" in adata.layers:
            return adata.layers["counts"]
        raise ValueError(
            f"This function requires count data, but neither `adata.X` nor "
            f"`adata.layers['counts']` in sdata.tables['{tables_key}'] "
            f"look available as counts."
        )

    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} does not exist in sdata.tables['{tables_key}'].layers.")

    X = adata.layers[layer]
    if _looks_like_counts(X):
        return X

    raise ValueError(
        f"Layer {layer!r} in sdata.tables['{tables_key}'] does not appear "
        f"to contain count data. This function requires non-negative integer counts."
    )
