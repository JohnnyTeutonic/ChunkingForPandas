"""example usage:
python cli.py chunk tests/data/sample.csv output.csv --strategy rows --n-chunks 3 --save-chunks
python cli.py benchmark tests/data/sample.csv --n-chunks 3 --save-chunks
"""
import click
import pandas as pd
from chunking_pandas.core import ChunkingExperiment, ChunkingStrategy, FileFormat
from chunking_pandas.visualization import plot_performance_comparison
from chunking_pandas.benchmark import run_benchmark

@click.group()
def cli():
    """Chunking for Pandas CLI tool."""
    pass

@cli.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--strategy', type=click.Choice([s.value for s in ChunkingStrategy]))
@click.option('--n-chunks', default=4)
@click.option('--save-chunks/--no-save-chunks', default=False)
def chunk(input_file, output_file, strategy, n_chunks, save_chunks):
    """Chunk a file using specified strategy."""
    experiment = ChunkingExperiment(
        input_file,
        output_file,
        chunking_strategy=strategy,
        n_chunks=n_chunks,
        save_chunks=save_chunks,
        monitor_performance=True
    )
    experiment.process_chunks(ChunkingStrategy(strategy))
    metrics = experiment.get_metrics()
    click.echo(f"Processing completed in {metrics[strategy].processing_time:.2f} seconds")

@cli.command()
@click.argument('input_file')
def benchmark(input_file):
    """Run benchmarking on input file."""
    results = run_benchmark(input_file, "benchmark_output.csv")
    click.echo(results) 