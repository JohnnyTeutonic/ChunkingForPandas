from chunking_experiment import ChunkingExperiment, ChunkingStrategy
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
if name == "main":
    main()
