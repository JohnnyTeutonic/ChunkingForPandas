import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def chunk_array_nb(data, n_chunks):
    """Optimized array chunking using Numba."""
    rows = data.shape[0]
    chunk_size = rows // n_chunks
    chunks = []
    
    for i in prange(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_chunks - 1 else rows
        chunks.append(data[start_idx:end_idx])
    
    return chunks

@jit(nopython=True)
def calculate_chunk_sizes(total_size, n_chunks):
    """Optimized chunk size calculation."""
    base_size = total_size // n_chunks
    remainder = total_size % n_chunks
    chunk_sizes = np.zeros(n_chunks, dtype=np.int64)
    
    for i in range(n_chunks):
        chunk_sizes[i] = base_size + (1 if i < remainder else 0)
    
    return chunk_sizes 