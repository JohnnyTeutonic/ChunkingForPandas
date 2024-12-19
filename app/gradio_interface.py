import gradio as gr
import pandas as pd
from pathlib import Path

import sys
from typing import Callable

# Add the project root to Python path to ensure imports work
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from chunking_pandas import ChunkingExperiment, FileFormat

def get_sample_data_path() -> Path:
    """Get the absolute path to sample data, checking multiple possible locations."""
    possible_paths = [
        Path(__file__).parent.parent / "tests" / "data" / "sample.csv",  # From app directory
        Path.cwd() / "tests" / "data" / "sample.csv",  # From project root
        Path(__file__).parent / "data" / "sample.csv",  # From app/data
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # If sample file doesn't exist, create it in the app/data directory
    default_path = Path(__file__).parent / "data" / "sample.csv"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not default_path.exists():
        # Create sample data
        df = pd.DataFrame({
            'A': range(100),
            'B': [f"Value_{i}" for i in range(100)],
            'C': ['X', 'Y', 'Z'] * 33 + ['X']
        })
        df.to_csv(default_path, index=False)
        print(f"Created sample data file at {default_path}")
    
    return default_path

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
            chunking_strategy=chunking_strategy,
            save_chunks=True
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

def create_interface():
    """Create and configure the Gradio interface."""
    sample_data_path = get_sample_data_path()
    
    return gr.Interface(
        fn=process_file,
        inputs=[
            gr.File(label="Input File"),
            gr.Textbox(label="Output Filename", placeholder="output.csv"),
            gr.Radio(
                choices=["csv", "json", "parquet", "numpy"],
                label="File Format",
                value="csv"
            ),
            gr.Radio(
                choices=["rows", "columns", "tokens", "blocks", "None"],
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
                str(sample_data_path),
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
    interface = create_interface()
    interface.launch(share=False, server_port=7860)

if __name__ == "__main__":
    launch_interface() 