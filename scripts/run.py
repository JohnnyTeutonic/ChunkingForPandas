import argparse
from chunking_pandas.core import ChunkingExperiment, ChunkingStrategy, FileFormat

def main():
    parser = argparse.ArgumentParser(description="Process files using various chunking strategies")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument("--n_chunks", type=int, default=4, help="Number of chunks to split the data into")
    parser.add_argument("--chunking_strategy", 
                       type=str, 
                       default="rows",
                       choices=[strategy.value for strategy in ChunkingStrategy],
                       help="Strategy to use for chunking data")
    parser.add_argument("--file_format", 
                       type=str, 
                       default="csv",
                       choices=[fmt.value for fmt in FileFormat],
                       help="Input file format")
    parser.add_argument("--save_chunks", 
                       action="store_true",
                       help="Whether to save individual chunks to disk")
    
    args = parser.parse_args()

    current_experiment = ChunkingExperiment(
        args.input, 
        args.output, 
        file_format=FileFormat(args.file_format), 
        n_chunks=args.n_chunks, 
        chunking_strategy=args.chunking_strategy,
        save_chunks=args.save_chunks,
        auto_run=False  # Prevent auto-running to have more control
    )
    current_experiment.process_chunks(ChunkingStrategy(args.chunking_strategy))

if __name__ == "__main__":
    main()