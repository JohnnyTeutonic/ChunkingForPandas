import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Callable
from chunking_experiment import ChunkingExperiment, ChunkingStrategy, FileFormat

def process_file(
    input_file, 
    output_filename: str,
    file_format: str,
    chunking_strategy: str,
    n_chunks: int
) -> Callable[[], tuple[gr.update, gr.update]]:
    """Process file using ChunkingExperiment and return paths to output files."""
    try:
        if input_file is None:
            raise ValueError("Please upload a file")
            
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
        
        return gr.update(value="\n".join(previews), visible=True), gr.update(visible=False)
    
    except Exception as e:
        # Return error message and reset interface
        return (
            gr.update(value="", visible=False),  # Clear and hide preview
            gr.update(value=f"Error: {str(e)}", visible=True)  # Show error message
        )

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
            choices=["rows", "columns", "tokens", "None"],
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
    outputs=[
        gr.Textbox(label="Output Preview", lines=10),
        gr.Textbox(label="Error Message", visible=False)
    ],
    title="File Chunking Interface",
    description="""
    Upload a file and specify how you want it chunked.
    The file will be split according to your specifications and saved as separate CSV files.
    A preview of each chunk will be shown below.
    """,
    examples=[
        [
            "../../tests/data/sample.csv",
            "output.csv",
            "csv",
            "rows",
            2
        ]
    ],
    allow_flagging="never"
)

def launch_interface():
    """Launch the Gradio interface."""
    iface.launch(share=False, server_port=7860)

if __name__ == "__main__":
    launch_interface() 