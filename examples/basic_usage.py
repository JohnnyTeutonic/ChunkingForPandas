from chunking_pandas.core import ChunkingExperiment
def main():
# Example usage of the ChunkingExperiment class
    experiment = ChunkingExperiment(
    "tests/data/sample.csv",
    "output.csv",
    n_chunks=3,
    chunking_strategy="rows",
    save_chunks=True
    )
# Show results
    print("Processing complete! Check output files.")
if __name__ == "__main__":
    main()
