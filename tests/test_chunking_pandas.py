import pytest
import pandas as pd
import numpy as np
from chunking_pandas.core import ChunkingExperiment, FileFormat, ChunkingStrategy

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'A': range(100),
        'B': [f"text_{i}" for i in range(100)],
        'C': [f"longer_text_{i*2}" for i in range(100)]
    })

@pytest.fixture
def sample_numpy_data():
    """Create sample numpy data for testing."""
    return np.random.rand(100, 4)

@pytest.fixture
def test_files(sample_data, sample_numpy_data, tmp_path):
    """Create test files in different formats."""
    csv_path = tmp_path / "test.csv"
    json_path = tmp_path / "test.json"
    parquet_path = tmp_path / "test.parquet"
    numpy_path = tmp_path / "test.npy"
    
    sample_data.to_csv(csv_path, index=False)
    sample_data.to_json(json_path)
    sample_data.to_parquet(parquet_path)
    np.save(numpy_path, sample_numpy_data)
    
    return {
        'csv': csv_path,
        'json': json_path,
        'parquet': parquet_path,
        'numpy': numpy_path
    }

def test_row_chunking(test_files, tmp_path):
    """Test row-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=4,
        chunking_strategy="rows",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    assert len(chunks) == 4
    
    # Check if output files exist
    for i in range(1, 5):
        chunk_file = tmp_path / f"output_chunk_{i}.csv"
        assert chunk_file.exists()
        
        # Check if each chunk has approximately equal size
        chunk_df = pd.read_csv(chunk_file)
        assert 20 <= len(chunk_df) <= 30

def test_column_chunking(test_files, tmp_path):
    """Test column-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=3,
        chunking_strategy="columns",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.COLUMNS)
    assert len(chunks) == 3
    
    # Check if output files exist and have correct number of columns
    for i in range(1, 4):
        chunk_file = tmp_path / f"output_chunk_{i}.csv"
        assert chunk_file.exists()
        chunk_df = pd.read_csv(chunk_file)
        assert 0 < len(chunk_df.columns) <= 1

def test_token_chunking(test_files, tmp_path):
    """Test token-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=2,
        chunking_strategy="tokens",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.TOKENS)
    assert len(chunks) == 2

def test_blocks_chunking(test_files, tmp_path):
    """Test blocks-based chunking strategy."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        n_chunks=4,
        chunking_strategy="blocks",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.BLOCKS)
    assert len(chunks) > 0

def test_numpy_chunking(test_files, tmp_path):
    """Test numpy array chunking."""
    output_file = tmp_path / "output.npy"
    experiment = ChunkingExperiment(
        str(test_files['numpy']),
        str(output_file),
        file_format=FileFormat.NUMPY,
        n_chunks=2,
        chunking_strategy="rows",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    assert len(chunks) == 2
    assert isinstance(chunks[0], np.ndarray)

def test_numpy_column_chunking(test_files, tmp_path):
    """Test numpy array column chunking."""
    output_file = tmp_path / "output.npy"
    experiment = ChunkingExperiment(
        str(test_files['numpy']),
        str(output_file),
        file_format=FileFormat.NUMPY,
        n_chunks=2,
        chunking_strategy="columns",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.COLUMNS)
    assert len(chunks) == 2

def test_numpy_blocks_chunking(test_files, tmp_path):
    """Test numpy array blocks chunking."""
    output_file = tmp_path / "output.npy"
    experiment = ChunkingExperiment(
        str(test_files['numpy']),
        str(output_file),
        file_format=FileFormat.NUMPY,
        n_chunks=4,
        chunking_strategy="blocks",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.BLOCKS)
    assert len(chunks) > 0

def test_no_chunks_strategy(test_files, tmp_path):
    """Test NO_CHUNKS strategy."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        chunking_strategy="None",
        save_chunks=True
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.NO_CHUNKS)
    assert len(chunks) == 1
    
    # Should contain all data
    original_df = pd.read_csv(test_files['csv'])
    assert len(chunks[0]) == len(original_df)

def test_auto_run_disabled(test_files, tmp_path):
    """Test with auto_run disabled."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        auto_run=False,
        save_chunks=True
    )
    
    # No files should exist yet
    chunk_file = tmp_path / "output_chunk_1.csv"
    assert not chunk_file.exists()

def test_save_chunks_disabled(test_files, tmp_path):
    """Test with save_chunks disabled."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        save_chunks=False
    )
    
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    # Files should not exist
    chunk_file = tmp_path / "output_chunk_1.csv"
    assert not chunk_file.exists()

def test_invalid_inputs(tmp_path):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        ChunkingExperiment(
            "nonexistent.csv",
            "output.csv",
            chunking_strategy="invalid_strategy"
        )
    
    with pytest.raises(ValueError):
        ChunkingExperiment(
            str(tmp_path / "test.txt"),
            "output.csv",
            file_format="txt"
        )
    
    # Create an empty json file to test format validation
    invalid_json = tmp_path / "test.json"
    invalid_json.touch()
    
    with pytest.raises(ValueError):
        ChunkingExperiment(
            str(invalid_json),
            "output.csv",
            file_format=FileFormat.JSON
        )

def test_numpy_invalid_strategies(test_files, tmp_path):
    """Test invalid strategies for numpy arrays."""
    output_file = tmp_path / "output.npy"
    experiment = ChunkingExperiment(
        str(test_files['numpy']),
        str(output_file),
        file_format=FileFormat.NUMPY
    )
    
    # Test invalid strategy
    with pytest.raises(ValueError):
        experiment.process_chunks(ChunkingStrategy.TOKENS)