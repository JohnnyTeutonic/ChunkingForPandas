"""
Example usage of optimized chunking functionality using Cython and Numba backends.
This example compares performance between different optimization strategies.
"""

import numpy as np
import pandas as pd
import time
from chunking_pandas import ChunkingExperiment
from chunking_pandas.visualization import plot_performance_comparison
import matplotlib.pyplot as plt

def compare_optimization_strategies():
    """Compare different optimization strategies for chunking."""
    # Create sample data
    data_size = 1_000_000
    n_features = 10
    data = np.random.rand(data_size, n_features)
    
    # Save as numpy file for testing
    np.save("large_array.npy", data)
    
    # Test different optimization backends
    backends = ['auto', 'cython', 'numba', None]
    results = {}
    
    for backend in backends:
        start_time = time.time()
        
        experiment = ChunkingExperiment(
            "large_array.npy",
            "output.npy",
            file_format="numpy",
            n_chunks=10,
            chunking_strategy="rows",
            use_optimized=backend is not None,
            optimization_backend=backend,
            monitor_performance=True
        )
        
        # Process chunks
        chunks = experiment.process_chunks("rows")
        
        # Get metrics
        metrics = experiment.get_metrics()
        results[f"backend_{backend}"] = metrics["rows"]
        
        print(f"\nBackend: {backend}")
        print(f"Processing time: {metrics['rows'].processing_time:.4f} seconds")
        print(f"Memory usage: {metrics['rows'].memory_usage:.2f} MB")
        print(f"Number of chunks: {metrics['rows'].total_chunks}")
        print(f"Average chunk size: {np.mean(metrics['rows'].chunk_sizes):.0f}")
    
    # Visualize results
    fig = plot_performance_comparison(results)
    plt.savefig('optimization_comparison.png')
    plt.close()

def demonstrate_parallel_processing():
    """Demonstrate parallel processing with optimized backends."""
    # Create large DataFrame
    df = pd.DataFrame(np.random.rand(500_000, 5), columns=[f'col_{i}' for i in range(5)])
    df.to_csv("large_df.csv", index=False)
    
    # Test parallel processing with different strategies
    strategies = ["parallel_rows", "parallel_blocks", "dynamic"]
    
    for strategy in strategies:
        experiment = ChunkingExperiment(
            "large_df.csv",
            "output.csv",
            chunking_strategy=strategy,
            n_chunks=4,
            n_workers=2,  # Use 2 workers for demonstration
            monitor_performance=True,
            use_optimized=True
        )
        
        chunks = experiment.process_chunks(strategy)
        metrics = experiment.get_metrics()
        
        print(f"\nStrategy: {strategy}")
        print(f"Processing time: {metrics[strategy].processing_time:.4f} seconds")
        print(f"Memory usage: {metrics[strategy].memory_usage:.2f} MB")
        print(f"Number of chunks: {metrics[strategy].total_chunks}")

if __name__ == "__main__":
    print("Comparing optimization strategies...")
    compare_optimization_strategies()
    
    print("\nDemonstrating parallel processing...")
    demonstrate_parallel_processing()
    
    print("\nResults have been saved to 'optimization_comparison.png'") 