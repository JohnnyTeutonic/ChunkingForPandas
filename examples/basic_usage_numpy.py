import numpy as np
from chunking_pandas import ChunkingExperiment, FileFormat, ChunkingStrategy

# Create a sample array
arr = np.random.rand(100, 100)
np.save('input.npy', arr)

# Create experiment
experiment = ChunkingExperiment(
    'input.npy',
    'output.npy',
    file_format=FileFormat.NUMPY,
    n_chunks=4,
    chunking_strategy='blocks'
)

# Process chunks
chunks = experiment.process_chunks(ChunkingStrategy.BLOCKS)