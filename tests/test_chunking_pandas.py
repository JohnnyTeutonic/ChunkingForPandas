import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
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
def sample_data_default():
    """Create sample data for default purposes."""
    return pd.DataFrame({'A': [1, 2, 3]})

@pytest.fixture
def sample_numpy_data():
    """Create sample numpy data for testing."""
    return np.random.rand(100, 4)

@pytest.fixture
def sample_1d_numpy_data():
    """Create 1D numpy data for testing."""
    return np.random.rand(100)

@pytest.fixture
def test_files(sample_data, sample_numpy_data, tmp_path):
    """Create test files in different formats."""
    files = {}
    for fmt in FileFormat:
        path = tmp_path / f"test.{fmt.value}"
        if fmt == FileFormat.CSV:
            sample_data.to_csv(path, index=False)
        elif fmt == FileFormat.JSON:
            sample_data.to_json(path)
        elif fmt == FileFormat.PARQUET:
            sample_data.to_parquet(path)
        elif fmt == FileFormat.NUMPY:
            np.save(path, sample_numpy_data)
        files[fmt.value] = path
    return files

# Test Enum Classes
def test_chunking_strategy_enum():
    """Test ChunkingStrategy enum values."""
    assert ChunkingStrategy.ROWS.value == "rows"
    assert ChunkingStrategy.COLUMNS.value == "columns"
    assert ChunkingStrategy.TOKENS.value == "tokens"
    assert ChunkingStrategy.BLOCKS.value == "blocks"
    assert ChunkingStrategy.NO_CHUNKS.value == "None"
    assert ChunkingStrategy.PARALLEL_ROWS.value == "parallel_rows"
    assert ChunkingStrategy.PARALLEL_BLOCKS.value == "parallel_blocks"
    assert ChunkingStrategy.DYNAMIC.value == "dynamic"

def test_file_format_enum():
    """Test FileFormat enum values."""
    assert FileFormat.CSV.value == "csv"
    assert FileFormat.JSON.value == "json"
    assert FileFormat.PARQUET.value == "parquet"
    assert FileFormat.NUMPY.value == "numpy"

# Test Initialization
def test_init_default_params(tmp_path, sample_data_default):
    """Test default initialization parameters."""
    # Create a temporary CSV file
    test_file = tmp_path / "test.csv"
    sample_data_default.to_csv(test_file, index=False)
    
    experiment = ChunkingExperiment(str(test_file), "output.csv", auto_run=False)
    assert experiment.file_format == FileFormat.CSV
    assert experiment.n_chunks == 4
    assert experiment.save_chunks is False
    assert experiment.n_workers == cpu_count()

def test_init_custom_params(test_files):
    """Test custom initialization parameters."""
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        "output.csv",
        file_format=FileFormat.CSV,
        n_chunks=10,
        save_chunks=True,
        n_workers=2,
        auto_run=False
    )
    assert experiment.n_chunks == 10
    assert experiment.save_chunks is True
    assert experiment.n_workers == 2

# Test File Format Validation
def test_file_format_validation(tmp_path, sample_data_default):
    """Test validation of file formats."""
    # Create test files with correct extensions
    csv_file = tmp_path / "test.csv"
    json_file = tmp_path / "test.json"
    parquet_file = tmp_path / "test.parquet"
    numpy_file = tmp_path / "test.npy"
    
    # Create sample data and save in different formats
    sample_data_default.to_csv(csv_file, index=False)
    sample_data_default.to_json(json_file)
    sample_data_default.to_parquet(parquet_file)
    np.save(numpy_file, sample_data_default.to_numpy())
    
    # Test each format
    format_files = {
        FileFormat.CSV: csv_file,
        FileFormat.JSON: json_file,
        FileFormat.PARQUET: parquet_file,
        FileFormat.NUMPY: numpy_file
    }
    
    for fmt, file_path in format_files.items():
        experiment = ChunkingExperiment(
            str(file_path),
            "output.csv",
            file_format=fmt,
            auto_run=False
        )
        assert experiment.file_format == fmt

    # Test invalid format
    wrong_ext_file = tmp_path / "test.txt"
    wrong_ext_file.touch()
    
    with pytest.raises(ValueError, match="Input file must be a .csv file"):
        ChunkingExperiment(
            str(wrong_ext_file),
            "output.csv",
            file_format=FileFormat.CSV
        )

# Test Chunking Strategies
@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunking_strategies(test_files, tmp_path, strategy):
    """Test all chunking strategies."""
    output_file = tmp_path / f"output_{strategy}.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        chunking_strategy=strategy,
        save_chunks=True,
        auto_run=False
    )
    chunks = experiment.process_chunks(strategy)
    assert len(chunks) > 0

# Test Parallel Processing
def test_parallel_processing(test_files, tmp_path):
    """Test parallel processing features."""
    for strategy in [ChunkingStrategy.PARALLEL_ROWS, ChunkingStrategy.PARALLEL_BLOCKS, ChunkingStrategy.DYNAMIC]:
        experiment = ChunkingExperiment(
            str(test_files['csv']),
            str(tmp_path / f"output_{strategy}.csv"),
            chunking_strategy=strategy,
            n_workers=2,
            auto_run=False
        )
        chunks = experiment.process_chunks(strategy)
        assert len(chunks) > 0

# Test numpy array processing
def test_numpy_array_processing(tmp_path):
    """Test processing of numpy arrays."""
    # Create sample numpy array and save it
    input_array = np.random.rand(10, 10)
    input_file = tmp_path / "test.npy"
    np.save(input_file, input_array)
    
    output_file = tmp_path / "output.npy"
    experiment = ChunkingExperiment(
        str(input_file),
        str(output_file),
        file_format=FileFormat.NUMPY,
        auto_run=False
    )
    
    for strategy in [ChunkingStrategy.ROWS, ChunkingStrategy.COLUMNS, ChunkingStrategy.BLOCKS]:
        chunks = experiment.process_chunks(strategy)
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

def test_numpy_1d_array_limitations(sample_1d_numpy_data, tmp_path):
    """Test 1D NumPy array limitations."""
    numpy_path = tmp_path / "test_1d.npy"
    np.save(numpy_path, sample_1d_numpy_data)
    experiment = ChunkingExperiment(
        str(numpy_path),
        str(tmp_path / "output.npy"),
        file_format=FileFormat.NUMPY,
        auto_run=False
    )
    
    # Should work with rows
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    assert len(chunks) > 0
    
    # Should fail with columns and blocks
    for strategy in [ChunkingStrategy.COLUMNS, ChunkingStrategy.BLOCKS]:
        with pytest.raises(ValueError):
            experiment.process_chunks(strategy)

# Test Memory Management
def test_optimal_chunk_size(tmp_path, sample_data_default):
    """Test optimal chunk size calculation."""
    # Create a temporary CSV file
    test_file = tmp_path / "test.csv"
    sample_data_default.to_csv(test_file, index=False)
    
    experiment = ChunkingExperiment(str(test_file), "output.csv", auto_run=False)
    sizes = [100, 1000, 10000]
    for size in sizes:
        chunk_size = experiment.get_optimal_chunk_size(size)
        assert chunk_size > 0
        assert chunk_size <= size

# Test Error Handling
def test_error_handling(tmp_path, sample_data_default):
    """Test error handling for invalid inputs."""
    # Create a valid CSV file for testing
    valid_csv = tmp_path / "test.csv"
    sample_data_default.to_csv(valid_csv, index=False)
    
    # Test invalid chunking strategy
    with pytest.raises(ValueError):
        ChunkingExperiment(
            str(valid_csv),
            "output.csv",
            chunking_strategy="invalid_strategy"
        )
    
    # Test invalid file format
    invalid_file = tmp_path / "test.txt"
    invalid_file.touch()
    
    with pytest.raises(ValueError):
        ChunkingExperiment(
            str(invalid_file),
            "output.csv",
            file_format=FileFormat.CSV
        )
    
    # Test nonexistent file with correct extension
    nonexistent = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        experiment = ChunkingExperiment(
            str(nonexistent),
            "output.csv",
            file_format=FileFormat.CSV
        )

# Test Saving Functionality
def test_save_chunks(test_files, tmp_path):
    """Test chunk saving functionality."""
    output_file = tmp_path / "output.csv"
    experiment = ChunkingExperiment(
        str(test_files['csv']),
        str(output_file),
        save_chunks=True,
        auto_run=False
    )
    chunks = experiment.process_chunks(ChunkingStrategy.ROWS)
    
    # Verify chunks are saved
    for i in range(1, len(chunks) + 1):
        chunk_file = tmp_path / f"output_chunk_{i}.csv"
        assert chunk_file.exists()

# Test Logging
def test_logging(tmp_path, caplog, sample_data_default):
    """Test logging functionality."""
    # Create a temporary CSV file
    test_file = tmp_path / "test.csv"
    sample_data_default.to_csv(test_file, index=False)
    
    experiment = ChunkingExperiment(str(test_file), "output.csv", save_chunks=False, auto_run=False)
    assert "Chunks will not be saved to disk" in caplog.text
