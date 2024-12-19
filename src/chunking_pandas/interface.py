import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Callable
from .core import ChunkingExperiment, FileFormat

def get_sample_data_path() -> Path:
    """Get the absolute path to sample data."""
    package_root = Path(__file__).parent
    sample_data_path = package_root / "data" / "sample.csv"
    
    # Create sample data if it doesn't exist
    if not sample_data_path.exists():
        sample_data_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'A': range(100),
            'B': [f"Value_{i}" for i in range(100)],
            'C': ['X', 'Y', 'Z'] * 33 + ['X']
        })
        df.to_csv(sample_data_path, index=False)
    
    return sample_data_path

def process_file(
    input_file, 
    output_filename: str,
    file_format: str,
    chunking_strategy: str,
    n_chunks: int
) -> tuple[gr.update, gr.update]:
    """Process file using ChunkingExperiment."""
    try:
        if input_file is None:
            raise ValueError("Please upload a file")
            
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        experiment = ChunkingExperiment(
            input_file.name,
            output_filename,
            file_format=FileFormat(file_format),
            n_chunks=n_chunks,
            chunking_strategy=chunking_strategy,
            save_chunks=True
        )
        
        output_base = output_filename.rsplit('.', 1)[0]
        output_paths = [
            f"{output_base}_chunk_{i+1}.csv" 
            for i in range(n_chunks)
        ]
        
        previews = []
        for path in output_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                preview = f"Preview of {path}:\n{df.head().to_string()}\n\n"
                previews.append(preview)
        
        return gr.update(value="\n".join(previews), visible=True), gr.update(visible=False)
    
    except Exception as e:
        return (
            gr.update(value="", visible=False),
            gr.update(value=f"Error: {str(e)}", visible=True)
        )

def create_interface() -> gr.Interface:
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
        examples=[[str(sample_data_path), "output.csv", "csv", "rows", 2]],
        allow_flagging="never"
    )

def launch_interface(share: bool = False, port: int = 7860):
    """Launch the Gradio interface."""
    interface = create_interface()
    interface.launch(share=share, server_port=port) 