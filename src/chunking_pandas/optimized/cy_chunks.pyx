# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from cython.parallel import prange

cpdef np.ndarray[double, ndim=2] chunk_array_cy(np.ndarray[double, ndim=2] data, int n_chunks):
    """Optimized array chunking using Cython."""
    cdef:
        int rows = data.shape[0]
        int cols = data.shape[1]
        int chunk_size = rows // n_chunks
        int i, j, chunk_idx
        np.ndarray[double, ndim=2] chunk
    
    chunks = []
    for chunk_idx in prange(n_chunks, nogil=True):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if chunk_idx < n_chunks - 1 else rows
        chunk = data[start_idx:end_idx]
        chunks.append(chunk)
    
    return chunks 