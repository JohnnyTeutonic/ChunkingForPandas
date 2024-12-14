from chunking_experiment.core import ChunkingExperiment, ChunkingStrategy
def main():
# Example usage of the ChunkingExperiment class
    experiment = ChunkingExperiment(
    "sample.csv",
    "output.csv",
    n_chunks=3,
    chunking_strategy="rows"
    )
# Show results
    print("Processing complete! Check output files.")
if __name__ == "__main__":
    main()
