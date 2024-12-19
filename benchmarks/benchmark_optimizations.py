import numpy as np
import pandas as pd
import time
from chunking_pandas.core import ChunkingExperiment, ChunkingStrategy

def benchmark_chunking(data_size=1000000, n_chunks=10):
    """Benchmark different chunking implementations."""
    # Create test data
    data = np.random.rand(data_size, 100)
    
    results = {}
    
    # Test different backends
    backends = ['default', 'numba', 'cython']
    for backend in backends:
        experiment = ChunkingExperiment(
            "test.npy",
            "output.npy",
            use_optimized=True,
            optimization_backend=backend
        )
        
        start_time = time.time()
        _ = experiment._optimize_chunks(data)
        end_time = time.time()
        
        results[backend] = end_time - start_time
    
    return results

if __name__ == "__main__":
    sizes = [100000, 1000000, 10000000]
    for size in sizes:
        print(f"\nBenchmarking with data size: {size}")
        results = benchmark_chunking(size)
        for backend, time_taken in results.items():
            print(f"{backend}: {time_taken:.4f} seconds") 