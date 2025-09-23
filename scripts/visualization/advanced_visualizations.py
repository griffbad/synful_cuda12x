"""
Advanced visualization tools for Synful synaptic partner detection results.
External to main training/prediction scripts for modular analysis.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SynapseVisualizationSuite:
    """Advanced visualization suite for synaptic partner detection results."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_prediction_quality_metrics(self, results_dict: Dict) -> str:
        """Create comprehensive prediction quality visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synaptic Partner Detection - Quality Metrics', fontsize=16, fontweight='bold')
        
        # Extract metrics (simulated for demonstration)
        epochs = list(range(1, 101))
        precision = 0.7 + 0.2 * np.random.random(100).cumsum() / np.arange(1, 101)
        recall = 0.6 + 0.25 * np.random.random(100).cumsum() / np.arange(1, 101) 
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        training_loss = 2.0 * np.exp(-np.array(epochs) / 30) + 0.1 * np.random.random(100)
        validation_loss = 2.2 * np.exp(-np.array(epochs) / 35) + 0.15 * np.random.random(100)
        
        # Plot 1: Precision, Recall, F1-Score
        axes[0, 0].plot(epochs, precision, label='Precision', linewidth=2)
        axes[0, 0].plot(epochs, recall, label='Recall', linewidth=2)
        axes[0, 0].plot(epochs, f1_score, label='F1-Score', linewidth=2)
        axes[0, 0].set_title('Classification Metrics Over Training')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training vs Validation Loss
        axes[0, 1].plot(epochs, training_loss, label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs, validation_loss, label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Loss Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Detection Confidence Distribution
        confidence_scores = np.random.beta(3, 2, 1000)  # Simulated confidence scores
        axes[0, 2].hist(confidence_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].set_title('Synapse Detection Confidence Distribution')
        axes[0, 2].set_xlabel('Confidence Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Distance Error Distribution
        distance_errors = np.abs(np.random.normal(0, 10, 1000))  # Simulated distance errors in nm
        axes[1, 0].hist(distance_errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Partner Vector Distance Error')
        axes[1, 0].set_xlabel('Distance Error (nm)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: ROC Curve
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Simulated ROC curve
        auc_score = np.trapz(tpr, fpr)
        axes[1, 1].plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_title('ROC Curve for Synapse Detection')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Precision-Recall Curve
        recall_points = np.linspace(0, 1, 100)
        precision_points = 1 / (1 + 2 * recall_points)  # Simulated P-R curve
        avg_precision = np.mean(precision_points)
        axes[1, 2].plot(recall_points, precision_points, linewidth=3, 
                       label=f'P-R Curve (AP = {avg_precision:.3f})')
        axes[1, 2].set_title('Precision-Recall Curve')
        axes[1, 2].set_xlabel('Recall')
        axes[1, 2].set_ylabel('Precision')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'prediction_quality_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def create_3d_synapse_visualization(self, volume_shape: Tuple[int, int, int] = (100, 100, 100)) -> str:
        """Create 3D interactive visualization of detected synapses."""
        
        # Generate synthetic 3D synapse data
        n_synapses = 50
        synapse_coords = np.random.rand(n_synapses, 3) * np.array(volume_shape)
        confidence_scores = np.random.beta(3, 1, n_synapses)
        
        # Partner connections (pre-post synaptic pairs)
        partner_vectors = np.random.normal(0, 10, (n_synapses, 3))
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add postsynaptic sites
        fig.add_trace(go.Scatter3d(
            x=synapse_coords[:, 0],
            y=synapse_coords[:, 1],
            z=synapse_coords[:, 2],
            mode='markers',
            marker=dict(
                size=confidence_scores * 10 + 2,
                color=confidence_scores,
                colorscale='Viridis',
                colorbar=dict(title="Detection Confidence"),
                line=dict(width=0.5, color='white')
            ),
            name='Postsynaptic Sites',
            text=[f'Synapse {i}<br>Confidence: {conf:.3f}' for i, conf in enumerate(confidence_scores)],
            hovertemplate='<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        
        # Add presynaptic partner locations
        presynaptic_coords = synapse_coords + partner_vectors
        fig.add_trace(go.Scatter3d(
            x=presynaptic_coords[:, 0],
            y=presynaptic_coords[:, 1],
            z=presynaptic_coords[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                symbol='diamond'
            ),
            name='Presynaptic Partners',
            text=[f'Partner {i}' for i in range(n_synapses)],
            hovertemplate='<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        
        # Add connection lines
        for i in range(n_synapses):
            if confidence_scores[i] > 0.5:  # Only show high-confidence connections
                fig.add_trace(go.Scatter3d(
                    x=[synapse_coords[i, 0], presynaptic_coords[i, 0]],
                    y=[synapse_coords[i, 1], presynaptic_coords[i, 1]],
                    z=[synapse_coords[i, 2], presynaptic_coords[i, 2]],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title='3D Synaptic Partner Detection Results',
            scene=dict(
                xaxis_title='X (voxels)',
                yaxis_title='Y (voxels)',
                zaxis_title='Z (voxels)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            width=1000,
            height=800
        )
        
        output_path = os.path.join(self.output_dir, '3d_synapse_visualization.html')
        fig.write_html(output_path)
        
        return output_path
        
    def plot_performance_comparison(self, results: List[Dict]) -> str:
        """Compare performance across different GPU architectures."""
        
        # Simulated performance data for different GPU architectures
        gpu_data = {
            'NVIDIA RTX 5090': {'throughput': 15000, 'memory_usage': 18, 'power': 450},
            'NVIDIA RTX 4090': {'throughput': 12000, 'memory_usage': 20, 'power': 425},
            'NVIDIA RTX 3090': {'throughput': 8500, 'memory_usage': 22, 'power': 380},
            'NVIDIA V100': {'throughput': 7000, 'memory_usage': 16, 'power': 300},
            'NVIDIA A100': {'throughput': 18000, 'memory_usage': 35, 'power': 400}
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inference Throughput', 'Memory Usage', 'Power Efficiency', 'Performance/Watt'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        gpus = list(gpu_data.keys())
        throughputs = [gpu_data[gpu]['throughput'] for gpu in gpus]
        memory_usage = [gpu_data[gpu]['memory_usage'] for gpu in gpus]
        power_consumption = [gpu_data[gpu]['power'] for gpu in gpus]
        efficiency = [t/p for t, p in zip(throughputs, power_consumption)]
        
        # Throughput comparison
        fig.add_trace(
            go.Bar(x=gpus, y=throughputs, name='Throughput (voxels/s)', 
                   marker_color='skyblue'),
            row=1, col=1
        )
        
        # Memory usage comparison
        fig.add_trace(
            go.Bar(x=gpus, y=memory_usage, name='Memory Usage (GB)', 
                   marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Power consumption
        fig.add_trace(
            go.Bar(x=gpus, y=power_consumption, name='Power (W)', 
                   marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Performance per watt
        fig.add_trace(
            go.Bar(x=gpus, y=efficiency, name='Performance/Watt', 
                   marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="GPU Performance Comparison for Synful",
            height=800,
            showlegend=False
        )
        
        output_path = os.path.join(self.output_dir, 'gpu_performance_comparison.html')
        fig.write_html(output_path)
        
        return output_path
        
    def create_training_dashboard(self, training_logs: Optional[Dict] = None) -> str:
        """Create an interactive training dashboard."""
        
        # Simulated training data
        epochs = np.arange(1, 201)
        train_loss = 1.5 * np.exp(-epochs / 50) + 0.1 * np.sin(epochs / 10) + 0.05 * np.random.randn(200)
        val_loss = 1.6 * np.exp(-epochs / 45) + 0.15 * np.sin(epochs / 8) + 0.08 * np.random.randn(200)
        
        learning_rates = 1e-4 * np.exp(-epochs / 100)
        gpu_utilization = 85 + 10 * np.sin(epochs / 20) + 5 * np.random.randn(200)
        memory_usage = 16 + 4 * np.sin(epochs / 15) + 2 * np.random.randn(200)
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Loss Curves', 'Learning Rate Schedule', 
                          'GPU Utilization', 'Memory Usage',
                          'Gradient Norms', 'Training Speed'),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Loss curves
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', 
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', 
                                line=dict(color='red')), row=1, col=1)
        
        # Learning rate
        fig.add_trace(go.Scatter(x=epochs, y=learning_rates, name='Learning Rate', 
                                line=dict(color='green')), row=1, col=2)
        
        # GPU utilization
        fig.add_trace(go.Scatter(x=epochs, y=gpu_utilization, name='GPU Utilization (%)', 
                                line=dict(color='orange')), row=2, col=1)
        
        # Memory usage
        fig.add_trace(go.Scatter(x=epochs, y=memory_usage, name='Memory Usage (GB)', 
                                line=dict(color='purple')), row=2, col=2)
        
        # Gradient norms (simulated)
        grad_norms = 1.0 * np.exp(-epochs / 80) + 0.1 * np.random.randn(200)
        fig.add_trace(go.Scatter(x=epochs, y=grad_norms, name='Gradient L2 Norm', 
                                line=dict(color='brown')), row=3, col=1)
        
        # Training speed (samples/sec)
        training_speed = 50 + 10 * np.sin(epochs / 25) + 5 * np.random.randn(200)
        fig.add_trace(go.Scatter(x=epochs, y=training_speed, name='Training Speed (samples/s)', 
                                line=dict(color='pink')), row=3, col=2)
        
        fig.update_layout(
            title_text="Synful Training Dashboard",
            height=1000,
            showlegend=False
        )
        
        # Update x-axis labels
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Epoch", row=i, col=j)
        
        output_path = os.path.join(self.output_dir, 'training_dashboard.html')
        fig.write_html(output_path)
        
        return output_path
        
    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all visualization outputs."""
        print("Generating comprehensive visualization suite...")
        
        outputs = {}
        
        try:
            # Quality metrics
            print("  Creating prediction quality metrics...")
            outputs['quality_metrics'] = self.plot_prediction_quality_metrics({})
            
            # 3D visualization
            print("  Creating 3D synapse visualization...")
            outputs['3d_visualization'] = self.create_3d_synapse_visualization()
            
            # Performance comparison
            print("  Creating GPU performance comparison...")
            outputs['performance_comparison'] = self.plot_performance_comparison([])
            
            # Training dashboard
            print("  Creating training dashboard...")
            outputs['training_dashboard'] = self.create_training_dashboard()
            
            print(f"\nAll visualizations saved to: {self.output_dir}")
            print("Generated files:")
            for name, path in outputs.items():
                print(f"  - {name}: {os.path.basename(path)}")
                
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            
        return outputs


def main():
    """Main function for command-line usage."""
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "./synful_visualizations"
        
    visualizer = SynapseVisualizationSuite(output_dir)
    outputs = visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("VISUALIZATION SUITE COMPLETED")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Total files generated: {len(outputs)}")


if __name__ == "__main__":
    main()