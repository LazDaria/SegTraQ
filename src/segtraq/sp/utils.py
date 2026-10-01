from scipy import sparse
import numpy as np

def _binary_detection_matrix(X, idx):
    """Return a binary cell-by-gene detection matrix."""
    det = X[:, idx]

    if sparse.issparse(det):
        det = det.tocsr(copy=True)
        det.eliminate_zeros()
        det.data = np.ones(det.nnz, dtype=np.int64)
        return det

    return (np.asarray(det) > 0).astype(np.int64)