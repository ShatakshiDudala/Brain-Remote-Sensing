#"""
#Visualization utilities for brain-inspired remote sensing analysis
#Includes attention maps, segmentation overlays, and performance visualizations
#"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2
from scipy import ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_attention_map(image, model_type='snn', attention_weights=None, 
                        layer_idx=-1, head_idx=0):
    """
    Create attention visualization for different model types
    
    Args:
        image: Input image
        model_type: Type of model ('snn', 'transformer', 'gradcam', 'saliency')
        attention_weights: Pre-computed attention weights
        layer_idx: Layer index for attention
        head_idx: Attention head index
    
    Returns:
        Attention map overlay
    """
    
    if model_type.lower() == 'snn':
        # SNN attention based on spike activity
        if attention_weights is not None:
            attention = attention_weights
        else:
            # Simulate spike-based attention
            attention = simulate_spike_attention(image)
        
        # Resize to image dimensions
        attention_resized = cv2.resize(attention, (image.shape[1], image.shape[0]))
    
    elif model_type.lower() == 'transformer':
        # Transformer self-attention
        if attention_weights is not None:
            # Use provided attention weights
            if len(attention_weights) > abs(layer_idx):
                layer_attention = attention_weights[layer_idx]
                if len(layer_attention.shape) == 4:  # (batch, heads, seq, seq)
                    attention = layer_attention[0, head_idx, 0, 1:]  # Class token attention
                else:
                    attention = layer_attention[0, 1:]  # Skip class token
                
                # Reshape to spatial dimensions
                patch_size = int(np.sqrt(len(attention)))
                attention = attention.reshape(patch_size, patch_size)
                
                # Resize to image dimensions
                attention_resized = cv2.resize(attention, (image.shape[1], image.shape[0]))
            else:
                attention_resized = np.ones((image.shape[0], image.shape[1])) * 0.5
        else:
            # Simulate transformer attention
            attention_resized = simulate_transformer_attention(image)
    
    elif model_type.lower() == 'gradcam':
        # Grad-CAM visualization
        attention_resized = simulate_gradcam(image)
    
    elif model_type.lower() == 'saliency':
        # Saliency map
        attention_resized = simulate_saliency(image)
    
    else:
        # Default: uniform attention
        attention_resized = np.ones((image.shape[0], image.shape[1])) * 0.5
    
    # Normalize attention map
    attention_resized = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
    
    return attention_resized

def simulate_spike_attention(image):
    """Simulate spike-based attention pattern"""
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Edge detection (spikes often occur at edges)
    edges = cv2.Canny(gray, 50, 150)
    
    # Gaussian blur to create attention field
    attention = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 5)
    
    # Add some randomness (temporal nature of spikes)
    noise = np.random.exponential(0.3, attention.shape)
    attention = attention * (1 + noise)
    
    return attention

def simulate_transformer_attention(image):
    """Simulate transformer-like attention pattern"""
    # Focus on central regions and high-contrast areas
    h, w = image.shape[:2]
    
    # Central bias
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    central_attention = np.exp(-distance_from_center / (min(h, w) / 4))
    
    # High-contrast areas
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Local standard deviation (measure of contrast)
    contrast = ndimage.generic_filter(gray.astype(np.float32), np.std, size=9)
    contrast_attention = contrast / (contrast.max() + 1e-8)
    
    # Combine central bias and contrast
    attention = 0.6 * central_attention + 0.4 * contrast_attention
    
    return attention

def simulate_gradcam(image):
    """Simulate Grad-CAM heatmap"""
    # Focus on salient features
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Use Sobel filters to detect features
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of gradients
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Smooth the gradients
    attention = cv2.GaussianBlur(magnitude, (21, 21), 7)
    
    return attention

def simulate_saliency(image):
    """Simulate saliency map"""
    # Spectral residual approach (simplified)
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # FFT
    f = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    
    # Spectral residual
    log_amplitude_blurred = cv2.GaussianBlur(log_amplitude, (3, 3), 1)
    spectral_residual = log_amplitude - log_amplitude_blurred
    
    # Inverse FFT
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase)))
    
    # Post-processing
    saliency = cv2.GaussianBlur(saliency, (11, 11), 2.5)
    
    return saliency

def create_segmentation_overlay(image, predictions=None, model_type='cnn', 
                              class_names=None, alpha=0.6):
    """
    Create segmentation overlay visualization
    
    Args:
        image: Original image
        predictions: Segmentation predictions
        model_type: Type of model
        class_names: Names of classes
        alpha: Overlay transparency
    
    Returns:
        Overlay visualization
    """
    if predictions is None:
        # Generate mock predictions
        predictions = generate_mock_segmentation(image.shape[:2])
    
    # Ensure predictions are 2D
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    # Create color map
    num_classes = len(np.unique(predictions))
    colors = create_class_colormap(num_classes)
    
    # Convert predictions to RGB
    pred_rgb = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    for class_id in np.unique(predictions):
        mask = predictions == class_id
        pred_rgb[mask] = colors[class_id % len(colors)]
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    # Resize prediction to match image if necessary
    if pred_rgb.shape[:2] != image_uint8.shape[:2]:
        pred_rgb = cv2.resize(pred_rgb, (image_uint8.shape[1], image_uint8.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image_uint8, 1-alpha, pred_rgb, alpha, 0)
    
    return overlay

def generate_mock_segmentation(shape, num_classes=5):
    """Generate mock segmentation for demonstration"""
    h, w = shape
    segmentation = np.zeros((h, w), dtype=np.int32)
    
    # Create some regions
    # Water (class 0)
    cv2.ellipse(segmentation, (w//4, h//4), (w//8, h//12), 0, 0, 360, 0, -1)
    
    # Vegetation (class 1)
    cv2.rectangle(segmentation, (w//2, h//3), (3*w//4, 2*h//3), 1, -1)
    
    # Urban (class 2)
    cv2.rectangle(segmentation, (w//8, 2*h//3), (w//3, 7*h//8), 2, -1)
    
    # Agriculture (class 3)
    cv2.ellipse(segmentation, (3*w//4, 3*h//4), (w//6, h//8), 0, 0, 360, 3, -1)
    
    # Add some noise
    noise = np.random.randint(0, num_classes, (h//8, w//8))
    segmentation[::8, ::8] = noise
    
    return segmentation

def create_class_colormap(num_classes):
    """Create colormap for different classes"""
    # Define colors for common remote sensing classes
    base_colors = [
        [0, 0, 0],        # Background/Unknown - Black
        [0, 100, 255],    # Water - Blue
        [34, 139, 34],    # Vegetation - Green
        [255, 0, 0],      # Urban - Red
        [139, 69, 19],    # Bare Soil - Brown
        [255, 255, 0],    # Agriculture - Yellow
        [128, 128, 128],  # Roads - Gray
        [255, 192, 203],  # Snow/Ice - Pink
        [160, 32, 240],   # Wetland - Purple
        [255, 165, 0],    # Desert - Orange
    ]
    
    # Extend with additional colors if needed
    while len(base_colors) < num_classes:
        # Generate random colors
        base_colors.append([
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ])
    
    return base_colors[:num_classes]

def plot_attention_comparison(image, attention_maps, model_names, 
                            save_path=None, figsize=(15, 5)):
    """
    Plot comparison of attention maps from different models
    
    Args:
        image: Original image
        attention_maps: List of attention maps
        model_names: List of model names
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    num_models = len(attention_maps)
    fig, axes = plt.subplots(1, num_models + 1, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention maps
    for i, (attention, name) in enumerate(zip(attention_maps, model_names)):
        # Overlay attention on image
        axes[i+1].imshow(image)
        im = axes[i+1].imshow(attention, alpha=0.6, cmap='hot', interpolation='bilinear')
        axes[i+1].set_title(f'{name} Attention')
        axes[i+1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_performance_comparison(results_dict, metrics=['accuracy', 'f1_score'], 
                              save_path=None, figsize=(12, 6)):
    """
    Plot performance comparison across models
    
    Args:
        results_dict: Dictionary of model results
        metrics: Metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    models = list(results_dict.keys())
    
    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            if 'metrics' in results_dict[model] and metric in results_dict[model]['metrics']:
                values.append(results_dict[model]['metrics'][metric])
            else:
                values.append(0)
        
        # Create bar plot
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 8:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(cm, class_names=None, normalize=False, 
                         save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Whether to normalize values
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_interactive_attention_plot(image, attention_map, title="Attention Visualization"):
    """
    Create interactive attention visualization using Plotly
    
    Args:
        image: Original image
        attention_map: Attention weights
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Original Image', 'Attention Map', 'Overlay'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Original image
    fig.add_trace(
        go.Heatmap(z=image[::-1] if len(image.shape) == 2 else image[::-1, :, 0],
                  colorscale='gray', showscale=False),
        row=1, col=1
    )
    
    # Attention map
    fig.add_trace(
        go.Heatmap(z=attention_map[::-1], colorscale='hot', showscale=True),
        row=1, col=2
    )
    
    # Overlay (simplified)
    overlay_data = image[::-1, :, 0] if len(image.shape) == 3 else image[::-1]
    fig.add_trace(
        go.Heatmap(z=overlay_data, colorscale='gray', showscale=False, opacity=0.7),
        row=1, col=3
    )
    fig.add_trace(
        go.Heatmap(z=attention_map[::-1], colorscale='hot', showscale=False, opacity=0.5),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    # Remove axis ticks
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, row=1, col=i)
    
    return fig

def create_model_comparison_radar(results_dict, metrics=None):
    """
    Create radar chart for model comparison
    
    Args:
        results_dict: Dictionary of model results
        metrics: Metrics to include in radar chart
    
    Returns:
        Plotly figure
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for model_name, results in results_dict.items():
        values = []
        for metric in metrics:
            if 'metrics' in results and metric in results['metrics']:
                values.append(results['metrics'][metric])
            else:
                values.append(0)
        
        # Close the radar chart
        values.append(values[0])
        metrics_display = metrics + [metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_display,
            fill='toself',
            name=model_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison"
    )
    
    return fig

def create_energy_efficiency_plot(results_dict):
    """
    Create energy efficiency visualization
    
    Args:
        results_dict: Dictionary of model results
    
    Returns:
        Plotly figure
    """
    models = list(results_dict.keys())
    accuracies = []
    energies = []
    
    for model in models:
        if 'metrics' in results_dict[model]:
            acc = results_dict[model]['metrics'].get('accuracy', 0)
        else:
            acc = 0
        
        energy = results_dict[model].get('energy_consumption', 100)
        
        accuracies.append(acc * 100)  # Convert to percentage
        energies.append(energy)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=energies,
        y=accuracies,
        mode='markers+text',
        text=models,
        textposition="top center",
        marker=dict(
            size=15,
            color=accuracies,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Accuracy (%)")
        ),
        name="Models"
    ))
    
    fig.update_layout(
        title="Model Accuracy vs Energy Consumption",
        xaxis_title="Energy Consumption (mJ)",
        yaxis_title="Accuracy (%)",
        height=500
    )
    
    # Add ideal region annotation
    fig.add_annotation(
        x=min(energies) * 1.1,
        y=max(accuracies) * 0.95,
        text="Ideal Region<br>(Low Energy, High Accuracy)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        font=dict(color="green")
    )
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    print("Visualization Utilities - Example Usage")
    
    # Create synthetic image
    print("\n1. Creating synthetic test data...")
    test_image = np.random.rand(128, 128, 3)
    
    # Test attention map creation
    print("\n2. Testing attention map creation...")
    
    attention_snn = create_attention_map(test_image, model_type='snn')
    attention_transformer = create_attention_map(test_image, model_type='transformer')
    attention_gradcam = create_attention_map(test_image, model_type='gradcam')
    
    print(f"SNN attention shape: {attention_snn.shape}")
    print(f"Transformer attention shape: {attention_transformer.shape}")
    print(f"Grad-CAM attention shape: {attention_gradcam.shape}")
    
    # Test segmentation overlay
    print("\n3. Testing segmentation overlay...")
    
    overlay = create_segmentation_overlay(test_image, model_type='cnn')
    print(f"Segmentation overlay shape: {overlay.shape}")
    
    # Test performance comparison
    print("\n4. Testing performance visualization...")
    
    mock_results = {
        'CNN': {
            'metrics': {
                'accuracy': 0.912,
                'precision': 0.903,
                'recall': 0.887,
                'f1_score': 0.895
            },
            'energy_consumption': 156.2
        },
        'SNN': {
            'metrics': {
                'accuracy': 0.932,
                'precision': 0.925,
                'recall': 0.911,
                'f1_score': 0.918
            },
            'energy_consumption': 23.4
        },
        'Transformer': {
            'metrics': {
                'accuracy': 0.948,
                'precision': 0.945,
                'recall': 0.937,
                'f1_score': 0.941
            },
            'energy_consumption': 145.7
        }
    }
    
    # Create interactive plots
    radar_fig = create_model_comparison_radar(mock_results)
    energy_fig = create_energy_efficiency_plot(mock_results)
    
    print("Interactive plots created successfully!")
    
    # Test confusion matrix
    print("\n5. Testing confusion matrix visualization...")
    
    # Create mock confusion matrix
    cm = np.array([
        [45, 2, 1, 0, 2],
        [3, 38, 1, 3, 5],
        [0, 1, 42, 4, 3],
        [1, 2, 2, 41, 4],
        [2, 3, 2, 3, 40]
    ])
    
    class_names = ['Water', 'Vegetation', 'Urban', 'Bare Soil', 'Agriculture']
    
    print(f"Confusion matrix shape: {cm.shape}")
    print("Class names:", class_names)
    
    print("\nVisualization utilities ready!")
    print("Key features:")
    print("- Attention map visualization")
    print("- Segmentation overlays")
    print("- Performance comparisons")
    print("- Interactive plots with Plotly")
    print("- Confusion matrix heatmaps")
    print("- Energy efficiency analysis")
    print("- Radar charts for model comparison")
    print("- Multi-model attention comparison")