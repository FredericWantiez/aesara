import numpy as np

import aesara
import scipy as sp
from aesara.sparse.sharedvar import SparseTensorSharedVariable


def test_shared_basic():
    x = aesara.shared(
        sp.sparse.csr_matrix(np.eye(100), dtype=np.float64), name="blah", borrow=True
    )

    assert isinstance(x, SparseTensorSharedVariable)
    assert x.format == "csr"
    assert x.dtype == "float64"
