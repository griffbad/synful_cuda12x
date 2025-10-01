"""
Modern CLI interface for Synful using Typer.
"""

import typer
from typing import Optional, List
from pathlib import Path
import torch
import lightning as L
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import hydra
from omegaconf import DictConfig
import wandb

from .training import SynfulTrainer
from .inference import SynfulPredictor
from .evaluation import SynapticPartnerEvaluator
from .visualization import SynfulVisualizer

app = typer.Typer(
    name="synful",
    help="Modern Synful: Synaptic partner detection in 3D electron microscopy",
    add_completion=False
)

console = Console()


@app.command()
def train(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to training configuration file"),
    data_dir: Path = typer.Option(..., "--data", "-d", help="Path to training data directory"),
    output_dir: Path = typer.Option("./outputs", "--output", "-o", help="Output directory for models and logs"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs to use"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb", help="Weights & Biases project name"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Train a Synful model with modern PyTorch Lightning."""
    
    console.print("[bold blue]Starting Synful Training[/bold blue]")
    
    # Load configuration
    with console.status("Loading configuration..."):
        config = hydra.compose(config_name=config_path.stem, overrides=[])
    
    # Initialize wandb if specified
    if wandb_project:
        wandb.init(
            project=wandb_project,
            config=config,
            name=f"synful_{config.model.name}_{config.experiment.name}"
        )
        console.print(f"[green]Initialized W&B project: {wandb_project}[/green]")
    
    # Create trainer
    trainer = SynfulTrainer(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        gpus=gpus,
        debug=debug
    )
    
    # Start training
    console.print("[yellow]Starting training...[/yellow]")
    trainer.train(resume_checkpoint=resume)
    
    console.print("[bold green]Training completed![/bold green]")


@app.command()
def predict(
    model_path: Path = typer.Option(..., "--model", "-m", help="Path to trained model checkpoint"),
    input_data: Path = typer.Option(..., "--input", "-i", help="Path to input data (HDF5/Zarr)"),
    output_dir: Path = typer.Option("./predictions", "--output", "-o", help="Output directory for predictions"),
    chunk_size: List[int] = typer.Option([64, 512, 512], "--chunk-size", help="Chunk size for processing [z,y,x]"),
    overlap: List[int] = typer.Option([8, 64, 64], "--overlap", help="Overlap between chunks [z,y,x]"),
    batch_size: int = typer.Option(1, "--batch-size", help="Batch size for inference"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cpu/cuda)"),
    save_raw: bool = typer.Option(False, "--save-raw", help="Save raw prediction outputs"),
    threshold: float = typer.Option(0.5, "--threshold", help="Threshold for mask predictions"),
):
    """Run inference on new data using a trained model."""
    
    console.print("[bold blue]Starting Synful Prediction[/bold blue]")
    
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[yellow]Auto-detected device: {device}[/yellow]")
    
    # Create predictor
    predictor = SynfulPredictor(
        model_path=model_path,
        device=device,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size
    )
    
    # Run prediction
    with console.status("Running prediction..."):
        results = predictor.predict(
            input_data=input_data,
            output_dir=output_dir,
            save_raw=save_raw,
            threshold=threshold
        )
    
    # Display results summary
    table = Table(title="Prediction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Input shape", str(results["input_shape"]))
    table.add_row("Output files", str(len(results["output_files"])))
    table.add_row("Processing time", f"{results['processing_time']:.1f}s")
    table.add_row("Detected synapses", str(results["num_synapses"]))
    
    console.print(table)
    console.print(f"[bold green]Predictions saved to: {output_dir}[/bold green]")


@app.command()
def evaluate(
    predictions: Path = typer.Option(..., "--predictions", "-p", help="Path to prediction results"),
    ground_truth: Path = typer.Option(..., "--ground-truth", "-gt", help="Path to ground truth annotations"),
    output_dir: Path = typer.Option("./evaluation", "--output", "-o", help="Output directory for evaluation results"),
    matching_threshold: float = typer.Option(400.0, "--threshold", help="Distance threshold for matching (nm)"),
    metrics: List[str] = typer.Option(["fscore", "precision", "recall"], "--metrics", help="Metrics to compute"),
    visualize: bool = typer.Option(True, "--visualize", help="Generate visualization plots"),
):
    """Evaluate predictions against ground truth annotations."""
    
    console.print("[bold blue]Starting Synful Evaluation[/bold blue]")
    
    # Create evaluator
    evaluator = SynapticPartnerEvaluator(
        matching_threshold=matching_threshold,
        metrics=metrics
    )
    
    # Run evaluation
    with console.status("Computing evaluation metrics..."):
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            output_dir=output_dir
        )
    
    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for metric, value in results["metrics"].items():
        table.add_row(metric.capitalize(), f"{value:.3f}")
    
    console.print(table)
    
    # Generate visualizations
    if visualize:
        with console.status("Generating visualizations..."):
            viz = SynfulVisualizer()
            viz.create_evaluation_plots(results, output_dir)
        
        console.print(f"[green]Visualizations saved to: {output_dir}[/green]")
    
    console.print(f"[bold green]Evaluation completed![/bold green]")


@app.command()
def visualize(
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to data file (synapses/predictions)"),
    output_dir: Path = typer.Option("./visualizations", "--output", "-o", help="Output directory for plots"),
    plot_type: str = typer.Option("all", "--type", "-t", help="Type of plot (all/distribution/scores/3d)"),
    interactive: bool = typer.Option(True, "--interactive", help="Generate interactive plots"),
    napari: bool = typer.Option(False, "--napari", help="Open in Napari viewer"),
):
    """Generate visualizations for synapse data and predictions."""
    
    console.print("[bold blue]Starting Synful Visualization[/bold blue]")
    
    # Create visualizer
    viz = SynfulVisualizer()
    
    # Load data
    with console.status("Loading data..."):
        if data_path.suffix == '.h5':
            from .synapse import SynapseCollection
            data = SynapseCollection.load_hdf5(data_path)
        elif data_path.suffix == '.zarr':
            from .synapse import SynapseCollection
            data = SynapseCollection.load_zarr(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots based on type
    if plot_type in ["all", "distribution"]:
        console.print("Generating distribution plots...")
        fig = viz.plot_synapse_distribution(
            data, 
            save_path=output_dir / "distribution.html",
            interactive=interactive
        )
    
    if plot_type in ["all", "scores"]:
        console.print("Generating score analysis...")
        fig = viz.plot_score_distribution(
            data,
            save_path=output_dir / "scores.html"
        )
    
    if plot_type in ["all", "summary"]:
        console.print("Generating summary report...")
        report = viz.create_summary_report(
            data,
            save_path=output_dir / "summary.html"
        )
    
    # Open in Napari if requested
    if napari:
        console.print("Opening in Napari...")
        viewer = viz.create_napari_viewer(
            raw_data=None,  # Would need to load raw data
            synapses=data
        )
        viewer.show()
    
    console.print(f"[bold green]Visualizations saved to: {output_dir}[/bold green]")


@app.command()
def info():
    """Display information about the Synful installation and environment."""
    
    console.print("[bold blue]Synful Environment Information[/bold blue]")
    
    # Create info table
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="magenta")
    
    # Python and PyTorch info
    import sys
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Lightning", L.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        table.add_row("GPU Names", ", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    
    # Package versions
    try:
        import gunpowder
        table.add_row("Gunpowder", gunpowder.__version__)
    except ImportError:
        table.add_row("Gunpowder", "[red]Not installed[/red]")
    
    try:
        import daisy
        table.add_row("Daisy", daisy.__version__)
    except ImportError:
        table.add_row("Daisy", "[red]Not installed[/red]")
    
    try:
        import napari
        table.add_row("Napari", napari.__version__)
    except ImportError:
        table.add_row("Napari", "[red]Not installed[/red]")
    
    console.print(table)


@app.command()
def convert_legacy(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Path to legacy Synful data"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory for converted data"),
    format: str = typer.Option("zarr", "--format", help="Output format (zarr/hdf5)"),
):
    """Convert legacy Synful data to modern format."""
    
    console.print("[bold blue]Converting Legacy Synful Data[/bold blue]")
    
    # This would implement conversion from old format to new
    console.print("[yellow]Legacy conversion not yet implemented[/yellow]")
    console.print("This feature will convert:")
    console.print("• Old synapse annotations to modern format")
    console.print("• TensorFlow models to PyTorch")
    console.print("• Legacy configuration files")


if __name__ == "__main__":
    app()