from enum import Enum
import logging
import pandas as pd
import numpy as np
from typing import Union, List

from .utils.logging import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

class ChunkingStrategy(str, Enum):
    ROWS = "rows"
    COLUMNS = "columns"
    TOKENS = "tokens"
    BLOCKS = "blocks"
    NO_CHUNKS = "None"

class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NUMPY = "numpy"

class ChunkingExperiment:
    def __init__(self, input_file: str, output_file: str, 
                 file_format: FileFormat = FileFormat.CSV, 
                 auto_run: bool = True, n_chunks: int = 4, 
                 chunking_strategy: str = "rows",
                 save_chunks: bool = False):
        """Initialize ChunkingExperiment with specified file format.
        
        Args:
            input_file: Path to input file or NumPy array
            output_file: Path to output file base name
            file_format: FileFormat enum specifying input file type
            auto_run: Whether to automatically run processing
            n_chunks: Number of chunks to split the data into
            chunking_strategy: Strategy to use for chunking data
            save_chunks: Whether to save chunks to disk (default: True)
        """
        self.file_format = file_format
        self.save_chunks = save_chunks
        
        if not save_chunks:
            logger.warning("Chunks will not be saved to disk as save_chunks=False")
        
        match file_format:
            case FileFormat.CSV:
                self.input_file = input_file
                self.output_file = output_file
            case FileFormat.JSON:
                if not input_file.endswith('.json'):
                    logger.error(f"Input file must be a JSON file, got: {input_file}")
                    raise ValueError(f"Input file must be a JSON file, got: {input_file}")
                self.input_file = input_file
                self.output_file = output_file
            case FileFormat.PARQUET:
                if not input_file.endswith('.parquet'):
                    logger.error(f"Input file must be a parquet file, got: {input_file}")
                    raise ValueError(f"Input file must be a parquet file, got: {input_file}")
                self.input_file = input_file
                self.output_file = output_file
            case FileFormat.NUMPY:
                if not input_file.endswith('.npy'):
                    logger.error(f"Input file must be a NumPy file, got: {input_file}")
                    raise ValueError(f"Input file must be a NumPy file, got: {input_file}")
                self.input_file = input_file
                self.output_file = output_file
            case _:
                logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")
        
        self.n_chunks = max(1, n_chunks)  # Ensure at least 1 chunk
        
        if auto_run:
            self.process_chunks(ChunkingStrategy(chunking_strategy))
    
    def _chunk_numpy_array(self, arr: np.ndarray, strategy: ChunkingStrategy) -> List[np.ndarray]:
        """Helper method to chunk NumPy arrays."""
        match strategy:
            case ChunkingStrategy.ROWS:
                chunk_size = arr.shape[0] // self.n_chunks
                return [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)]
                
            case ChunkingStrategy.COLUMNS:
                if arr.ndim < 2:
                    raise ValueError("Cannot chunk 1D array by columns")
                chunk_size = arr.shape[1] // self.n_chunks
                return [arr[:, i:i + chunk_size] for i in range(0, arr.shape[1], chunk_size)]
                
            case ChunkingStrategy.BLOCKS:
                if arr.ndim < 2:
                    raise ValueError("Cannot chunk 1D array into blocks")
                rows, cols = arr.shape
                block_rows = int(rows ** 0.5)
                block_cols = int(cols ** 0.5)
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
        """Process input data into chunks and optionally save them to output files.
        
        Args:
            strategy: ChunkingStrategy enum specifying how to split the data
            
        Returns:
            List of pandas DataFrames or NumPy arrays representing the chunks
        """
        # Read the input file
        file_extension = self.input_file.split('.')[-1].lower()
        match file_extension:
            case "csv":
                df = pd.read_csv(self.input_file)
                is_numpy = False
            case "json":
                df = pd.read_json(self.input_file)
                is_numpy = False
            case "parquet":
                df = pd.read_parquet(self.input_file)
                is_numpy = False
            case "npy":
                df = np.load(self.input_file)
                is_numpy = True
            case _:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        # Get output file base name without extension
        output_base = self.output_file.rsplit('.', 1)[0]
        output_ext = self.output_file.rsplit('.', 1)[1]

        if is_numpy:
            chunks = self._chunk_numpy_array(df, strategy)
            # Save NumPy chunks only if save_chunks is True
            if self.save_chunks:
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{output_base}_chunk_{i+1}.npy"
                    np.save(chunk_filename, chunk)
                    logger.info(f"Saved NumPy chunk {i+1} to {chunk_filename}")
            return chunks

        # Process pandas DataFrame chunks
        chunks = []
        match strategy:
            case ChunkingStrategy.ROWS:
                chunk_size = len(df) // self.n_chunks
                chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
                
            case ChunkingStrategy.COLUMNS:
                chunk_size = len(df.columns) // self.n_chunks
                chunks = [df.iloc[:, i:i + chunk_size] for i in range(0, len(df.columns), chunk_size)]
                
            case ChunkingStrategy.TOKENS:
                token_counts = df.astype(str).apply(lambda x: x.str.len().sum(), axis=1)
                total_tokens = token_counts.sum()
                tokens_per_chunk = total_tokens // self.n_chunks
                
                current_chunk_start = 0
                current_token_count = 0
                
                for idx, token_count in enumerate(token_counts):
                    current_token_count += token_count
                    if current_token_count >= tokens_per_chunk and len(chunks) < self.n_chunks - 1:
                        chunks.append(df.iloc[current_chunk_start:idx + 1])
                        current_chunk_start = idx + 1
                        current_token_count = 0
                
                if current_chunk_start < len(df):
                    chunks.append(df.iloc[current_chunk_start:])
                
            case ChunkingStrategy.BLOCKS:
                rows = len(df)
                cols = len(df.columns)
                block_rows = int(rows ** 0.5)
                block_cols = int(cols ** 0.5)
                
                for i in range(0, rows, block_rows):
                    for j in range(0, cols, block_cols):
                        block = df.iloc[i:min(i + block_rows, rows), 
                                      j:min(j + block_cols, cols)]
                        chunks.append(block)
                
            case ChunkingStrategy.NO_CHUNKS:
                chunks = [df]
                
            case _:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Save pandas DataFrame chunks only if save_chunks is True
        if self.save_chunks:
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{output_base}_chunk_{i+1}.{output_ext}"
                chunk.to_csv(chunk_filename, index=False)
                logger.info(f"Saved chunk {i+1} to {chunk_filename}")

        return chunks