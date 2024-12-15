Usage
=====

Basic Usage
----------

.. code-block:: python

    from chunking_experiment import ChunkingExperiment, ChunkingStrategy, FileFormat

    # Create an experiment
    experiment = ChunkingExperiment(
        "input.csv",
        "output.csv",
        n_chunks=3,
        chunking_strategy="rows"
    )

Web Interface
------------

To run the web interface:

.. code-block:: python

    from chunking_experiment.gradio_interface import launch_interface
    launch_interface()

Chunking Strategies
-----------------

The package supports several chunking strategies:

* **rows**: Split data by rows
* **columns**: Split data by columns
* **tokens**: Split data by approximate token count
* **blocks**: Split data into block matrices (for 2D data)
* **none**: Keep data as single chunk

File Formats
-----------

Supported file formats:

* CSV
* JSON
* Parquet
* NumPy arrays (.npy) 