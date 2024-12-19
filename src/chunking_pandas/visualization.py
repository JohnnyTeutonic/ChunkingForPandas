import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from chunking_pandas.core import ChunkingMetrics

def plot_performance_comparison(metrics: Dict[str, ChunkingMetrics]):
    """Plot performance metrics comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Processing time comparison
    strategies = list(metrics.keys())
    times = [m.processing_time for m in metrics.values()]
    sns.barplot(x=strategies, y=times, ax=ax1)
    ax1.set_title('Processing Time by Strategy')
    ax1.set_ylabel('Time (seconds)')
    
    # Memory usage comparison
    memory = [m.memory_usage for m in metrics.values()]
    sns.barplot(x=strategies, y=memory, ax=ax2)
    ax2.set_title('Memory Usage by Strategy')
    ax2.set_ylabel('Memory (MB)')
    
    # Chunk size distribution
    for strategy, metric in metrics.items():
        sns.kdeplot(data=metric.chunk_sizes, label=strategy, ax=ax3)
    ax3.set_title('Chunk Size Distribution')
    ax3.set_xlabel('Chunk Size')
    
    plt.tight_layout()
    return fig 