# Chunking for Pandas

[![codecov](https://codecov.io/gh/JohnnyTeutonic/ChunkingForPandas/branch/main/graph/badge.svg)](https://codecov.io/gh/JohnnyTeutonic/ChunkingForPandas)
[![Tests](https://github.com/JohnnyTeutonic/ChunkingForPandas/actions/workflows/test.yml/badge.svg)](https://github.com/JohnnyTeutonic/ChunkingForPandas/actions/workflows/test.yml)

A Python package to chunk pandas/numpy data with different chunking strategies.

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

Or:

```bash
make install
```

## Usage

```python
from chunking_pandas import ChunkingExperiment
```

## Create an instance of a Chunking class

```python
class_instance = ChunkingExperiment(
"input.csv",
"output.csv",
n_chunks=3,
chunking_strategy="rows"
)
```

Then perform the chunking:

```python
class_instance.process_chunks()
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
- Benchmarking the chunking strategies

## Development

To install development dependencies:

```bash
pip install -e .[dev]
```

Or:

```bash
make install-dev
```

## Testing

To run tests, run the following command from the root folder:

```bash
pytest
```

## Documentation

To install the documentation dependencies:

```bash
pip install -e .[docs]
```

Or:

```bash
make install-docs
```

To build the documentation:

```bash
make docs
```

To serve the documentation:

```bash
make docs-serve
```

Below are the full list of commands that can be run from the root folder:

```bash
make benchmark
make clean
make docs
make docs-clean
make docs-serve
make install
make install-docs
make lint
make run
make test
make typecheck
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
