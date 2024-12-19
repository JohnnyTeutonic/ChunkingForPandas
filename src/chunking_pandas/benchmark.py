import pandas as pd
import numpy as np
from typing import Dict, List
from chunking_pandas.core import ChunkingExperiment, ChunkingStrategy, FileFormat

def run_benchmark(input_file: str, output_file: str,
                 strategies: List[ChunkingStrategy] = None,
                 sizes: List[int] = None) -> pd.DataFrame:
    """Run benchmarking for different strategies and data sizes."""
    if strategies is None:
        strategies = list(ChunkingStrategy)
    if sizes is None:
        sizes = [1000, 10000, 100000]
    
    results = []
    for size in sizes:
        # Create test data
        df = pd.DataFrame(np.random.rand(size, 10))
        test_file = f"test_{size}.csv"
        df.to_csv(test_file, index=False)
        
        for strategy in strategies:
            experiment = ChunkingExperiment(
                test_file,
                output_file,
                monitor_performance=True,
                chunking_strategy=strategy,
                auto_run=False
            )
            chunks = experiment.process_chunks(strategy)
            metrics = experiment.get_metrics()[str(strategy)]
            
            results.append({
                'strategy': str(strategy),
                'data_size': size,
                'processing_time': metrics.processing_time,
                'memory_usage': metrics.memory_usage,
                'n_chunks': metrics.total_chunks
            })
    
    return pd.DataFrame(results) 