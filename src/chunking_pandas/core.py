from enum import Enum
import logging
import pandas as pd
import numpy as np
from typing import Union, List, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, Any

from .utils.logging import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

class ChunkingStrategy(str, Enum):
    ROWS = "rows"
    COLUMNS = "columns"
    TOKENS = "tokens"
    BLOCKS = "blocks"
    NO_CHUNKS = "None"
    PARALLEL_ROWS = "parallel_rows"
    PARALLEL_BLOCKS = "parallel_blocks"
    DYNAMIC = "dynamic"

class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"

@dataclass
class ChunkingMetrics:
    """Store metrics about chunking operations."""
    processing_time: float
    memory_usage: float
    chunk_sizes: List[int]
    strategy: str
    total_chunks: int

class ChunkingExperiment:
    def __init__(self, input_file: str, output_file: str, 
                 file_format: FileFormat = FileFormat.CSV, 
                 auto_run: bool = True, n_chunks: int = 4, 
                 chunking_strategy: str = "rows",
                 save_chunks: bool = False,
                 n_workers: Optional[int] = None,
                 monitor_performance: bool = False):
        """Initialize ChunkingExperiment with specified file format."""
        if n_workers is not None and n_workers <= 0:
            raise ValueError("Number of workers must be positive")
            
        self.file_format = file_format
        self.save_chunks = save_chunks
        self.output_file = output_file
        self.n_workers = n_workers or cpu_count()
        
        if not save_chunks:
            logger.warning("Chunks will not be saved to disk as save_chunks=False")
        
        # Check if input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file does not exist: {input_file}")
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Map file formats to their expected extensions
        format_extensions = {
            FileFormat.CSV: '.csv',
            FileFormat.JSON: '.json',
            FileFormat.PARQUET: '.parquet',
            FileFormat.NUMPY: '.npy'
        }
        
        expected_ext = format_extensions[file_format]
        if not input_file.lower().endswith(expected_ext):
            logger.error(f"Input file must be a {expected_ext} file, got: {input_file}")
            raise ValueError(f"Input file must be a {expected_ext} file, got: {input_file}")
        
        self.input_file = input_file
        self.n_chunks = max(1, n_chunks)
        
        if auto_run:
            self.process_chunks(ChunkingStrategy(chunking_strategy))
        
        self.monitor_performance = monitor_performance
        self.metrics: Dict[str, ChunkingMetrics] = {}

    def _read_input_file(self) -> Union[pd.DataFrame, np.ndarray]:
        """Read input file based on file format."""
        file_extension = self.input_file.split('.')[-1].lower()
        match file_extension:
            case "csv":
                return pd.read_csv(self.input_file)
            case "json":
                return pd.read_json(self.input_file)
            case "parquet":
                return pd.read_parquet(self.input_file)
            case "npy":
                return np.load(self.input_file)
            case _:
                raise ValueError(f"Unsupported file extension: {file_extension}")

    def _process_chunk_parallel(self, chunk_data: tuple) -> Union[pd.DataFrame, np.ndarray]:
        """Process a single chunk in parallel."""
        data, start_idx, end_idx = chunk_data
        return data.iloc[start_idx:end_idx] if isinstance(data, pd.DataFrame) else data[start_idx:end_idx]

    def _process_block_parallel(self, block_data: tuple) -> Union[pd.DataFrame, np.ndarray]:
        """Process a single block in parallel."""
        data, start_row, end_row, start_col, end_col = block_data
        if isinstance(data, pd.DataFrame):
            return data.iloc[start_row:end_row, start_col:end_col]
        return data[start_row:end_row, start_col:end_col]

    def _chunk_parallel(self, data: Union[pd.DataFrame, np.ndarray], strategy: ChunkingStrategy) -> List[Union[pd.DataFrame, np.ndarray]]:
        """Parallel chunking implementation."""
        total_size = len(data)
        chunk_size = max(1, total_size // self.n_chunks)
        
        if strategy == ChunkingStrategy.PARALLEL_ROWS:
            chunk_args = [(data, i, min(i + chunk_size, total_size)) 
                         for i in range(0, total_size, chunk_size)]
            with Pool(self.n_workers) as pool:
                return pool.map(self._process_chunk_parallel, chunk_args)
        
        elif strategy == ChunkingStrategy.PARALLEL_BLOCKS:
            rows = len(data)
            cols = len(data.columns) if isinstance(data, pd.DataFrame) else data.shape[1]
            block_rows = max(1, int(rows ** 0.5))
            block_cols = max(1, int(cols ** 0.5))
            
            block_args = [
                (data, i, min(i + block_rows, rows), j, min(j + block_cols, cols))
                for i in range(0, rows, block_rows)
                for j in range(0, cols, block_cols)
            ]
            
            with Pool(self.n_workers) as pool:
                return pool.map(self._process_block_parallel, block_args)
        
        elif strategy == ChunkingStrategy.DYNAMIC:
            # Dynamic chunking based on data size and available CPU cores
            optimal_chunk_size = max(1, total_size // (self.n_workers * 2))
            chunk_args = [(data, i, min(i + optimal_chunk_size, total_size))
                         for i in range(0, total_size, optimal_chunk_size)]
            
            with Pool(self.n_workers) as pool:
                return pool.map(self._process_chunk_parallel, chunk_args)

    def _chunk_numpy_array(self, arr: np.ndarray, strategy: ChunkingStrategy) -> List[np.ndarray]:
        """Helper method to chunk NumPy arrays."""
        match strategy:
            case ChunkingStrategy.ROWS:
                chunk_size = max(1, arr.shape[0] // self.n_chunks)  # Ensure chunk_size is at least 1
                return [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)]
                
            case ChunkingStrategy.COLUMNS:
                if arr.ndim < 2:
                    raise ValueError("Cannot chunk 1D array by columns")
                chunk_size = max(1, arr.shape[1] // self.n_chunks)  # Ensure chunk_size is at least 1
                return [arr[:, i:i + chunk_size] for i in range(0, arr.shape[1], chunk_size)]
                
            case ChunkingStrategy.BLOCKS:
                if arr.ndim < 2:
                    raise ValueError("Cannot chunk 1D array into blocks")
                rows, cols = arr.shape
                block_rows = max(1, int(rows ** 0.5))  # Ensure block size is at least 1
                block_cols = max(1, int(cols ** 0.5))
                chunks = []
                for i in range(0, rows, block_rows):
                    for j in range(0, cols, block_cols):
                        block = arr[i:min(i + block_rows, rows), 
                                  j:min(j + block_cols, cols)]
                        chunks.append(block)
                return chunks
                
            case ChunkingStrategy.NO_CHUNKS:
                return [arr]
                
            case _:
                raise ValueError(f"Unsupported chunking strategy for NumPy arrays: {strategy}")

    def process_chunks(self, strategy: ChunkingStrategy) -> Union[List[pd.DataFrame], List[np.ndarray]]:
        """Process input data into chunks with parallel support and performance monitoring."""
        start_time = time.time() if hasattr(self, "monitor_performance") and self.monitor_performance else None
        initial_memory = self._get_memory_usage() if hasattr(self, "monitor_performance") and self.monitor_performance else None
        data = self._read_input_file()
        is_numpy = isinstance(data, np.ndarray)
        
        # Handle parallel strategies
        if strategy in [ChunkingStrategy.PARALLEL_ROWS, ChunkingStrategy.PARALLEL_BLOCKS, ChunkingStrategy.DYNAMIC]:
            chunks = self._chunk_parallel(data, strategy)
        elif is_numpy:
            chunks = self._chunk_numpy_array(data, strategy)
        else:
            # Process pandas DataFrame chunks
            chunks = []
            match strategy:
                case ChunkingStrategy.ROWS:
                    chunk_size = max(1, len(data) // self.n_chunks)
                    chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                    
                case ChunkingStrategy.COLUMNS:
                    chunk_size = max(1, len(data.columns) // self.n_chunks)
                    chunks = [data.iloc[:, i:i + chunk_size] for i in range(0, len(data.columns), chunk_size)]
                    
                case ChunkingStrategy.TOKENS:
                    token_counts = data.astype(str).apply(lambda x: x.str.len().sum(), axis=1)
                    total_tokens = token_counts.sum()
                    tokens_per_chunk = max(1, total_tokens // self.n_chunks)
                    
                    current_chunk_start = 0
                    current_token_count = 0
                    
                    for idx, token_count in enumerate(token_counts):
                        current_token_count += token_count
                        if current_token_count >= tokens_per_chunk and len(chunks) < self.n_chunks - 1:
                            chunks.append(data.iloc[current_chunk_start:idx + 1])
                            current_chunk_start = idx + 1
                            current_token_count = 0
                    
                    if current_chunk_start < len(data):
                        chunks.append(data.iloc[current_chunk_start:])
                    
                case ChunkingStrategy.BLOCKS:
                    rows = len(data)
                    cols = len(data.columns)
                    block_rows = max(1, int(rows ** 0.5))
                    block_cols = max(1, int(cols ** 0.5))
                    
                    for i in range(0, rows, block_rows):
                        for j in range(0, cols, block_cols):
                            block = data.iloc[i:min(i + block_rows, rows), 
                                          j:min(j + block_cols, cols)]
                            chunks.append(block)
                    
                case ChunkingStrategy.NO_CHUNKS:
                    chunks = [data]
                    
                case _:
                    raise ValueError(f"Unknown chunking strategy: {strategy}")

        if self.save_chunks:
            output_base = self.output_file.rsplit('.', 1)[0]
            output_ext = self.output_file.rsplit('.', 1)[1]
            
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{output_base}_chunk_{i+1}.{output_ext}"
                if isinstance(chunk, np.ndarray):
                    np.save(chunk_filename, chunk)
                else:
                    chunk.to_csv(chunk_filename, index=False)
                logger.info(f"Saved chunk {i+1} to {chunk_filename}")

        if hasattr(self, "monitor_performance") and self.monitor_performance:
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            chunk_sizes = [len(chunk) for chunk in chunks]
            
            self.metrics[str(strategy)] = ChunkingMetrics(
                processing_time=processing_time,
                memory_usage=final_memory - initial_memory,
                chunk_sizes=chunk_sizes,
                strategy=str(strategy),
                total_chunks=len(chunks)
            )
        
        return chunks

    def get_optimal_chunk_size(self, data_size: int) -> int:
        """Calculate optimal chunk size based on data size and available resources."""
        memory_per_row = 1000  # Approximate memory per row in bytes
        available_memory = 1024 * 1024 * 1024  # 1GB default limit
        
        # Calculate based on memory and CPU cores
        memory_based_size = available_memory // (memory_per_row * self.n_workers)
        cpu_based_size = max(1, data_size // (self.n_workers * 2))
        
        return min(memory_based_size, cpu_based_size)

    def get_metrics(self) -> Dict[str, ChunkingMetrics]:
        """Return performance metrics for all operations."""
        return self.metrics

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024