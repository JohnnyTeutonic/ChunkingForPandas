# Chunking Experiment

[![codecov](https://codecov.io/gh/JohnnyTeutonic/chunking_experiment/branch/main/graph/badge.svg?token=00000000-0000-0000-0000-000000000000)](https://codecov.io/gh/JohnnyTeutonic/chunking_experiment)
[![Tests](https://github.com/JohnnyTeutonic/ChunkingForPandas/actions/workflows/test.yml/badge.svg)](https://github.com/JohnnyTeutonic/ChunkingForPandas/actions/workflows/test.yml)

A Python package for experimenting with different data chunking strategies.

## Requirements

- Python 3.10+
- Gradio
- pandas
- pytest
- numpy

## Installation

```bash
pip install chunking-experiment
```

## Usage

```python
from chunking_experiment import ChunkingExperiment, ChunkingStrategy, FileFormat
```

## Create an instance of a Chunking class

```python
experiment = ChunkingExperiment(
"input.csv",
"output.csv",
n_chunks=3,
chunking_strategy="rows"
)
```

## Run the web interface

Go to the app folder and run the following command:

```python
from gradio_interface import launch_interface
launch_interface()
```

Alternatively, you can run the following command to start the web interface:

```bash
python gradio_interface.py
```

## Features

- Multiple chunking strategies (rows, columns, tokens)
- Support for CSV, JSON, Numpy and Parquet files
- Web interface using Gradio
- Comprehensive test suite
- Documentation using Sphinx

## Development

To install development dependencies:

```bash
pip install -e .[dev]
```

## Testing

To run tests:

```bash
pytest
```

## Documentation

To install the documentation dependencies:

```bash
pip install -e .[docs]
```

To build the documentation:

```bash
make docs
```

To serve the documentation:

```bash
make docs-serve
```

Below are the commands that can be run from the root folder:

```bash
make lint
make typecheck
make test
make clean
make docs
make docs-serve
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
