import gradio as gr
import pandas as pd
from pathlib import Path
from chunking_experiment import ChunkingExperiment, ChunkingStrategy, FileFormat

def process_file(
    input_file, 
    output_filename: str,
    file_format: str,
    chunking_strategy: str,
    n_chunks: int
) -> list[str]:
    """Process file using ChunkingExperiment and return paths to output files."""
    
    # Ensure output filename has .csv extension
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'
    
    # Create ChunkingExperiment instance
    experiment = ChunkingExperiment(
        input_file.name,
        output_filename,
        file_format=FileFormat(file_format),
        n_chunks=n_chunks,
        chunking_strategy=chunking_strategy
    )
    
    # Get output file paths
    output_base = output_filename.rsplit('.', 1)[0]
    output_paths = [
        f"{output_base}_chunk_{i+1}.csv" 
        for i in range(n_chunks)
    ]
    
    # Return preview of first few rows of each chunk
    previews = []
    for path in output_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            preview = f"Preview of {path}:\n{df.head().to_string()}\n\n"
            previews.append(preview)
    
    return "\n".join(previews)

# Create Gradio interface
iface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Input File"),
        gr.Textbox(label="Output Filename", placeholder="output.csv"),
        gr.Radio(
            choices=["csv", "json", "parquet"],
            label="File Format",
            value="csv"
        ),
        gr.Radio(
            choices=["rows", "columns", "tokens", "no_chunks"],
            label="Chunking Strategy",
            value="rows"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            label="Number of Chunks",
            value=2
        )
    ],
    outputs=gr.Textbox(label="Output Preview", lines=10),
    title="File Chunking Interface",
    description="""
    Upload a file and specify how you want it chunked.
    The file will be split according to your specifications and saved as separate CSV files.
    A preview of each chunk will be shown below.
    """,
    examples=[
        [
            "sample.csv",  # You'll need to provide sample files
            "output.csv",
            "csv",
            "rows",
            2
        ]
    ]
)

if __name__ == "__main__":
    iface.launch() 