"""
Modern visualization tools for Synful with interactive plots and 3D rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

try:
    import neuroglancer
    NEUROGLANCER_AVAILABLE = True
except ImportError:
    NEUROGLANCER_AVAILABLE = False

from ..synapse import Synapse, SynapseCollection


class SynfulVisualizer:
    """
    Modern visualization tools for Synful analysis and results.
    """
    
    def __init__(self, style: str = "darkgrid"):
        """Initialize visualizer with plotting style."""
        sns.set_style(style)
        self.colors = px.colors.qualitative.Set2
        
    def plot_synapse_distribution(
        self, 
        synapses: SynapseCollection,
        save_path: Optional[Path] = None,
        interactive: bool = True
    ) -> go.Figure:
        """
        Plot 3D distribution of synapses with interactive controls.
        """
        df = synapses.to_dataframe()
        
        if interactive:
            fig = px.scatter_3d(
                df,
                x='pre_x', y='pre_y', z='pre_z',
                color='score',
                size='confidence' if 'confidence' in df.columns else None,
                hover_data=['id', 'id_segm_pre', 'id_segm_post'],
                title="3D Synapse Distribution",
                color_continuous_scale="Viridis"
            )
            
            # Add post-synaptic sites
            if 'post_x' in df.columns:
                fig.add_trace(
                    go.Scatter3d(
                        x=df['post_x'],
                        y=df['post_y'], 
                        z=df['post_z'],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color='red',
                            opacity=0.6
                        ),
                        name='Post-synaptic',
                        hovertemplate='Post-synaptic<br>x: %{x}<br>y: %{y}<br>z: %{z}'
                    )
                )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="X (nm)",
                    yaxis_title="Y (nm)",
                    zaxis_title="Z (nm)"
                ),
                title_x=0.5,
                width=800,
                height=600
            )
        else:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                df['pre_x'], df['pre_y'], df['pre_z'],
                c=df['score'] if 'score' in df.columns else 'blue',
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            
            if 'post_x' in df.columns:
                ax.scatter(
                    df['post_x'], df['post_y'], df['post_z'],
                    c='red', alpha=0.6, s=10, label='Post-synaptic'
                )
            
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_title('3D Synapse Distribution')
            
            if 'score' in df.columns:
                plt.colorbar(scatter, label='Score')
            
            fig = plt.gcf()
        
        if save_path:
            if interactive:
                fig.write_html(str(save_path))
            else:
                fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                
        return fig
        
    def plot_score_distribution(
        self,
        synapses: SynapseCollection,
        bins: int = 50,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """Plot distribution of synapse scores."""
        df = synapses.to_dataframe()
        
        if 'score' not in df.columns:
            raise ValueError("No scores available in synapse collection")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Distribution', 'Score vs Confidence', 
                           'Score by Segment', 'Score Cumulative'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['score'], nbinsx=bins, name='Score Distribution'),
            row=1, col=1
        )
        
        # Score vs Confidence
        if 'confidence' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['score'], y=df['confidence'],
                    mode='markers',
                    name='Score vs Confidence',
                    opacity=0.6
                ),
                row=1, col=2
            )
        
        # Box plot by segment type
        if 'id_segm_pre' in df.columns:
            # Sample a few segments for visualization
            top_segments = df['id_segm_pre'].value_counts().head(10).index
            df_subset = df[df['id_segm_pre'].isin(top_segments)]
            
            fig.add_trace(
                go.Box(
                    x=df_subset['id_segm_pre'].astype(str),
                    y=df_subset['score'],
                    name='Score by Segment'
                ),
                row=2, col=1
            )
        
        # Cumulative distribution
        sorted_scores = np.sort(df['score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_scores, y=cumulative,
                mode='lines',
                name='Cumulative Distribution'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Synapse Score Analysis",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def plot_training_metrics(
        self,
        metrics_log: Union[str, Path, Dict[str, List[float]]],
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """Plot training metrics over time."""
        
        if isinstance(metrics_log, (str, Path)):
            # Load from file (assuming CSV format)
            df = pd.read_csv(metrics_log)
        elif isinstance(metrics_log, dict):
            df = pd.DataFrame(metrics_log)
        else:
            raise ValueError("metrics_log must be file path or dictionary")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Dice Score', 'IoU', 'Learning Rate'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss curves
        if 'train_loss' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['train_loss'],
                    name='Train Loss',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        if 'val_loss' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['val_loss'],
                    name='Val Loss',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # Dice score
        if 'dice' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['dice'],
                    name='Dice Score',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # IoU
        if 'iou' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['iou'],
                    name='IoU',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
        
        # Learning rate
        if 'lr' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['lr'],
                    name='Learning Rate',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Training Metrics",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def visualize_predictions(
        self,
        raw_data: np.ndarray,
        predictions: Dict[str, np.ndarray],
        slice_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Visualize model predictions alongside raw data.
        """
        if slice_idx is None:
            slice_idx = raw_data.shape[0] // 2
            
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Raw Data', 'Mask Prediction', 'Direction X',
                           'Direction Y', 'Direction Z', 'Combined'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Raw data
        fig.add_trace(
            go.Heatmap(
                z=raw_data[slice_idx],
                colorscale='Gray',
                name='Raw'
            ),
            row=1, col=1
        )
        
        # Mask prediction
        if 'mask' in predictions:
            fig.add_trace(
                go.Heatmap(
                    z=predictions['mask'][slice_idx],
                    colorscale='Viridis',
                    name='Mask'
                ),
                row=1, col=2
            )
        
        # Direction vectors
        if 'vectors' in predictions:
            vectors = predictions['vectors']
            
            # X component
            fig.add_trace(
                go.Heatmap(
                    z=vectors[slice_idx, 0],
                    colorscale='RdBu',
                    name='Direction X'
                ),
                row=1, col=3
            )
            
            # Y component
            fig.add_trace(
                go.Heatmap(
                    z=vectors[slice_idx, 1],
                    colorscale='RdBu',
                    name='Direction Y'
                ),
                row=2, col=1
            )
            
            # Z component
            fig.add_trace(
                go.Heatmap(
                    z=vectors[slice_idx, 2],
                    colorscale='RdBu',
                    name='Direction Z'
                ),
                row=2, col=2
            )
            
            # Vector magnitude
            magnitude = np.linalg.norm(vectors[slice_idx], axis=0)
            fig.add_trace(
                go.Heatmap(
                    z=magnitude,
                    colorscale='Hot',
                    name='Vector Magnitude'
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title_text=f"Model Predictions (Slice {slice_idx})",
            title_x=0.5,
            height=800
        )
        
        if save_path:
            fig.write_html(str(save_path))
            
        return fig
        
    def create_napari_viewer(
        self,
        raw_data: np.ndarray,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        synapses: Optional[SynapseCollection] = None
    ):
        """
        Create interactive Napari viewer for 3D data exploration.
        """
        if not NAPARI_AVAILABLE:
            raise ImportError("Napari not available. Install with: pip install napari")
            
        viewer = napari.Viewer()
        
        # Add raw data
        viewer.add_image(raw_data, name='Raw Data', colormap='gray')
        
        # Add predictions
        if predictions:
            if 'mask' in predictions:
                viewer.add_image(
                    predictions['mask'], 
                    name='Mask Prediction',
                    colormap='viridis',
                    opacity=0.7
                )
            
            if 'vectors' in predictions:
                viewer.add_image(
                    predictions['vectors'],
                    name='Direction Vectors',
                    colormap='turbo',
                    opacity=0.5
                )
        
        # Add synapses as points
        if synapses:
            df = synapses.to_dataframe()
            
            if 'pre_z' in df.columns:
                pre_points = df[['pre_z', 'pre_y', 'pre_x']].values
                viewer.add_points(
                    pre_points,
                    name='Pre-synaptic',
                    size=5,
                    face_color='red',
                    edge_color='darkred'
                )
            
            if 'post_z' in df.columns:
                post_points = df[['post_z', 'post_y', 'post_x']].values
                viewer.add_points(
                    post_points,
                    name='Post-synaptic', 
                    size=3,
                    face_color='blue',
                    edge_color='darkblue'
                )
        
        return viewer
        
    def create_summary_report(
        self,
        synapses: SynapseCollection,
        metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[Path] = None
    ) -> str:
        """
        Create comprehensive HTML summary report.
        """
        stats = synapses.get_statistics()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synful Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 8px; text-align: center; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Synful Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>{stats['total_synapses']}</h3>
                        <p>Total Synapses</p>
                    </div>
                    <div class="stat-box">
                        <h3>{stats['complete_pairs']}</h3>
                        <p>Complete Pairs</p>
                    </div>
                    <div class="stat-box">
                        <h3>{stats['with_scores']}</h3>
                        <p>With Scores</p>
                    </div>
                </div>
            </div>
        """
        
        if 'score_stats' in stats:
            score_stats = stats['score_stats']
            html_content += f"""
            <div class="section">
                <h2>Score Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean</td><td>{score_stats['mean']:.3f}</td></tr>
                    <tr><td>Standard Deviation</td><td>{score_stats['std']:.3f}</td></tr>
                    <tr><td>Minimum</td><td>{score_stats['min']:.3f}</td></tr>
                    <tr><td>Maximum</td><td>{score_stats['max']:.3f}</td></tr>
                    <tr><td>Median</td><td>{score_stats['median']:.3f}</td></tr>
                </table>
            </div>
            """
        
        if metrics:
            html_content += f"""
            <div class="section">
                <h2>Model Performance</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
            """
            for metric, value in metrics.items():
                html_content += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
        
        return html_content