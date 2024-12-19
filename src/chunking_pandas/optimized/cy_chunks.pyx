# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=True

import numpy as np
cimport numpy as np
np.import_array()

def chunk_array_cy(np.ndarray data, int n_chunks):
    """Safe Cython implementation of array chunking."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if data.size == 0:
        return [data]
        
    try:
        chunk_size = max(1, data.shape[0] // n_chunks)
        chunks = []
        
        for i in range(0, data.shape[0], chunk_size):
            end_idx = min(i + chunk_size, data.shape[0])
            # Use a safer way to create chunks
            chunk = data[i:end_idx].copy()
            chunks.append(chunk)
        
        return chunks
    except:
        # Return single chunk if anything goes wrong
        return [data.copy()]