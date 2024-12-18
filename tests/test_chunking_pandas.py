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
    
    with pytest.raises(KeyError):
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

def test_file_format_extensions(tmp_path):
    """Test all file format extensions."""
    format_extensions = {
        FileFormat.CSV: '.csv',
        FileFormat.JSON: '.json',
        FileFormat.PARQUET: '.parquet',
        FileFormat.NUMPY: '.npy'
    }
    
    for format_type, extension in format_extensions.items():
        # Create test file with uppercase extension
        test_file = tmp_path / f"test{extension.upper()}"
        test_file.touch()
        
        # Test with uppercase extension
        experiment = ChunkingExperiment(
            str(test_file),
            str(tmp_path / f"output{extension}"),
            file_format=format_type,
            auto_run=False
        )
        assert experiment.input_file.endswith(extension.upper())

def test_chunking_strategy_conversion(tmp_path, sample_data):
    """Test conversion of string chunking strategy to enum."""
    # Create a test CSV file
    test_file = tmp_path / "test.csv"
    sample_data.to_csv(test_file, index=False)
    
    experiment = ChunkingExperiment(
        str(test_file),
        str(tmp_path / "output.csv"),
        chunking_strategy="rows",
        auto_run=False
    )
    
    # Test all valid strategies
    for strategy in ["rows", "columns", "tokens", "blocks", "None"]:
        chunks = experiment.process_chunks(ChunkingStrategy(strategy))
        assert isinstance(chunks, list)

def test_numpy_array_edge_cases(sample_numpy_data, tmp_path):
    """Test edge cases for numpy array operations."""
    numpy_path = tmp_path / "test.npy"
    np.save(str(numpy_path), sample_numpy_data)
    
    experiment = ChunkingExperiment(
        str(numpy_path),
        str(tmp_path / "output.npy"),
        file_format=FileFormat.NUMPY,
        auto_run=False
    )
    
    # Test with n_chunks larger than array size
    experiment.n_chunks = 1000
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    assert len(chunks) > 0
    
    # Test with single chunk
    experiment.n_chunks = 1
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    assert len(chunks) == 1

def test_output_file_extensions(test_files, tmp_path):
    """Test different output file extensions."""
    for input_format in FileFormat:
        input_file = test_files[input_format.value]
        
        # Test with different output extensions
        for output_ext in ['.csv', '.json', '.parquet', '.npy']:
            output_file = tmp_path / f"output{output_ext}"
            experiment = ChunkingExperiment(
                str(input_file),
                str(output_file),
                file_format=input_format,
                save_chunks=True,
                auto_run=False
            )
            
            chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
            assert len(chunks) > 0
            
            # Check if chunk files use input format's extension
            chunk_file = tmp_path / f"output_chunk_1{output_ext}"
            assert chunk_file.exists()

def test_invalid_chunk_numbers():
    """Test invalid chunk numbers."""
    # Test with zero chunks (should be converted to 1)
    experiment = ChunkingExperiment(
        "test.csv",
        "output.csv",
        n_chunks=0,
        auto_run=False
    )
    assert experiment.n_chunks == 1
    
    # Test with negative chunks (should be converted to 1)
    experiment = ChunkingExperiment(
        "test.csv",
        "output.csv",
        n_chunks=-5,
        auto_run=False
    )
    assert experiment.n_chunks == 1

def test_unsupported_file_extension(tmp_path):
    """Test handling of unsupported file extensions."""
    # Create a file with unsupported extension
    test_file = tmp_path / "test.txt"
    test_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        experiment = ChunkingExperiment(
            str(test_file),
            "output.csv"
        )
    assert "Input file must be" in str(exc_info.value)
    
    # Test reading unsupported extension
    experiment = ChunkingExperiment(
        "test.csv",
        "output.csv",
        auto_run=False
    )
    experiment.input_file = str(test_file)
    with pytest.raises(ValueError) as exc_info:
        experiment.process_chunks(ChunkingStrategy.ROWS)
    assert "Unsupported file extension" in str(exc_info.value)