import argparse
from core import ChunkingExperiment, ChunkingStrategy, FileFormat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunking_strategy", type=str, default="rows")
    parser.add_argument("--file_format", type=str, default="csv")
    args = parser.parse_args()

    current_experiment = ChunkingExperiment(
        args.input, 
        args.output, 
        FileFormat(args.file_format), 
        n_chunks=args.n_chunks, 
        chunking_strategy=args.chunking_strategy
    )
    current_experiment.process_chunks(ChunkingStrategy(args.chunking_strategy))

if __name__ == "__main__":
    main()