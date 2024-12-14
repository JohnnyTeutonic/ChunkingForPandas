from enum import Enum
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy(str, Enum):
    ROWS = "rows"
    COLUMNS = "columns"
    TOKENS = "tokens"
    NO_CHUNKS = "None"

class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class ChunkingExperiment:
    def __init__(self, input_file: str, output_file: str, file_format: FileFormat = FileFormat.CSV, 
                 auto_run: bool = True, n_chunks: int = 1, chunking_strategy: str = "rows"):
        """Initialize CursorExperiment with specified file format.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file base name
            file_format: FileFormat enum specifying input file type
            auto_run: Whether to automatically run processing
            n_chunks: Number of chunks to split the data into
            chunking_strategy: Strategy to use for chunking data
        """
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
            case _:
                logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")
        
        self.n_chunks = max(1, n_chunks)  # Ensure at least 1 chunk
        
        if auto_run:
            self.process_chunks(ChunkingStrategy(chunking_strategy))
    
    def process_chunks(self, strategy: ChunkingStrategy) -> list[pd.DataFrame]:
        """Process input data into chunks and save them to output files.
        
        Args:
            strategy: ChunkingStrategy enum specifying how to split the data
            
        Returns:
            List of pandas DataFrames representing the chunks
        """
        # Read the input file
        file_extension = self.input_file.split('.')[-1].lower()
        match file_extension:
            case "csv":
                df = pd.read_csv(self.input_file)
            case "json":
                df = pd.read_json(self.input_file)
            case "parquet":
                df = pd.read_parquet(self.input_file)
            case _:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        # Get output file base name without extension
        output_base = self.output_file.rsplit('.', 1)[0]
        output_ext = self.output_file.rsplit('.', 1)[1]

        chunks = []
        match strategy:
            case ChunkingStrategy.ROWS:
                # Split into even-sized chunks by rows
                chunk_size = len(df) // self.n_chunks
                chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
                
            case ChunkingStrategy.COLUMNS:
                # Split into even-sized chunks by columns
                chunk_size = len(df.columns) // self.n_chunks
                chunks = [df.iloc[:, i:i + chunk_size] for i in range(0, len(df.columns), chunk_size)]
                
            case ChunkingStrategy.TOKENS:
                # Calculate token counts for each row
                token_counts = df.astype(str).apply(
                    lambda x: x.str.len().sum(), axis=1
                )
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
                
                # Add remaining rows to last chunk
                if current_chunk_start < len(df):
                    chunks.append(df.iloc[current_chunk_start:])
                
            case ChunkingStrategy.NO_CHUNKS:
                chunks = [df]
                
            case _:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Save each chunk to a separate file
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{output_base}_chunk_{i+1}.{output_ext}"
            chunk.to_csv(chunk_filename, index=False)
            logger.info(f"Saved chunk {i+1} to {chunk_filename}")

        return chunks


