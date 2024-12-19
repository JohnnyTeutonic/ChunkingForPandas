# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

def chunk_array_cy(np.ndarray[double, ndim=2] data, int n_chunks):
    """Safe Cython implementation of array chunking."""
    cdef:
        int rows = data.shape[0]
        int chunk_size = max(1, rows // n_chunks)
        list chunks = []
        int i, end_idx
        np.ndarray[double, ndim=2] chunk
    
    try:
        for i in range(0, rows, chunk_size):
            end_idx = min(i + chunk_size, rows)
            chunk = np.array(data[i:end_idx], copy=True)
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        chunk_size = max(1, rows // n_chunks)
        return [np.array(data[i:i + chunk_size], copy=True) 
                for i in range(0, rows, chunk_size)]