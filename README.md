# Chunking Experiment

A Python package for experimenting with different data chunking strategies.

## Installation

```bash
pip install chunking-experiment
```

## Usage

```python
from chunking_experiment import ChunkingExperiment, ChunkingStrategy, FileFormat
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Create an experiment

```python
experiment = ChunkingExperiment(
"input.csv",
"output.csv",
n_chunks=3,
chunking_strategy="rows"
)
```

## Create an Experiment

experiment = ChunkingExperiment(
    input_file="input.csv",
    output_file="output.csv",
    n_chunks=3,
    chunking_strategy="rows"
)


## Run the web interface

```python
from chunking_experiment.interface import launch_interface
launch_interface()
```

## Features

- Multiple chunking strategies (rows, columns, tokens)
- Support for CSV, JSON, and Parquet files
- Web interface using Gradio
- Comprehensive test suite

## Development

To install development dependencies:

```bash
pip install -e .[dev]
```

To run tests:

```bash
pytest
```
