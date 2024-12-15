"""
Benchmarking the chunking strategies.
"""

import logging
import timeit
import pandas as pd
import numpy as np
from chunking_pandas import setup_logging, ChunkingExperiment, ChunkingStrategy

setup_logging()

logger = logging.getLogger(__name__)

def benchmark_strategies():
    df = pd.DataFrame(np.random.rand(100000, 10))
    df.to_csv("test.csv", index=False)
    results = {}
    for strategy in ChunkingStrategy:
        if strategy == ChunkingStrategy.NO_CHUNKS:
            continue
        start_time = timeit.default_timer()
        experiment = ChunkingExperiment("test.csv", "output.csv", chunking_strategy=strategy)
        experiment.process_chunks(strategy)
        elapsed = timeit.default_timer() - start_time
        logger.info(f"Chunking strategy {strategy} took {elapsed} seconds")
        results[strategy] = elapsed
    
    return results

benchmark_strategies()