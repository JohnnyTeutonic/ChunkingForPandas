import pytest
import pandas as pd
from chunking_pandas.core import ChunkingExperiment, FileFormat

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'A': range(100),
        'B': [f"text_{i}" for i in range(100)],
        'C': [f"longer_text_{i*2}" for i in range(100)]
    })

@pytest.fixture
def test_files(sample_data, tmp_path):
    """Create test files in different formats."""
    csv_path = tmp_path / "test.csv"
    json_path = tmp_path / "test.json"
    parquet_path = tmp_path / "test.parquet"
    
    sample_data.to_csv(csv_path, index=False)
    sample_data.to_json(json_path)
    sample_data.to_parquet(parquet_path)
    
    return {
        'csv': csv_path,
        'json': json_path,
        'parquet': parquet_path
    }

def test_row_chunking(test_files, tmp_path):
    """Test row-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=4,
        chunking_strategy="rows",
        save_chunks=True
    )
    
    # Check if output files exist
    for i in range(1, 5):
        chunk_file = tmp_path / f"output_chunk_{i}.csv"
        assert chunk_file.exists()
        
        # Check if each chunk has approximately equal size
        chunk_df = pd.read_csv(chunk_file)
        assert 20 <= len(chunk_df) <= 30  # Allow some flexibility in chunk size

def test_column_chunking(test_files, tmp_path):
    """Test column-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=3,
        chunking_strategy="columns",
        save_chunks=True

    )
    
    # Check if output files exist and have correct number of columns
    for i in range(1, 4):
        chunk_file = tmp_path / f"output_chunk_{i}.csv"
        assert chunk_file.exists()
        
        chunk_df = pd.read_csv(chunk_file)
        assert 0 < len(chunk_df.columns) <= 1  # Each chunk should have 1 column

def test_token_chunking(test_files, tmp_path):
    """Test token-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=2,
        chunking_strategy="tokens",
        save_chunks=True
    )
    
    # Check if output files exist
    chunk_files = [tmp_path / f"output_chunk_{i}.csv" for i in range(1, 3)]
    for file in chunk_files:
        assert file.exists()

def test_file_formats(test_files, tmp_path):
    """Test different input file formats."""
    for format_type, input_file in test_files.items():
        output_file = tmp_path / f"output_{format_type}.csv"
        
        ChunkingExperiment(
            str(input_file),
            str(output_file),
            file_format=FileFormat(format_type),
            n_chunks=2,
            chunking_strategy="rows",
            save_chunks=True
        )
        
        # Check if output files exist for each format
        for i in range(1, 3):
            chunk_file = tmp_path / f"output_{format_type}_chunk_{i}.csv"
            assert chunk_file.exists()

def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        ChunkingExperiment(
            "nonexistent.csv",
            "output.csv",
            chunking_strategy="invalid_strategy"
        )
    
    with pytest.raises(ValueError):
        ChunkingExperiment(
            "test.txt",  # Unsupported format
            "output.csv",
            file_format="txt"
        )

def test_no_chunks(test_files, tmp_path):
    """Test NO_CHUNKS strategy."""
    output_file = tmp_path / "output.csv"
    ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        chunking_strategy="None",
        save_chunks=True
    )
    
    # Should only create one output file
    chunk_file = tmp_path / "output_chunk_1.csv"
    assert chunk_file.exists()
    
    # Should contain all data
    original_df = pd.read_csv(test_files['csv'])
    chunk_df = pd.read_csv(chunk_file)
    assert len(original_df) == len(chunk_df) 