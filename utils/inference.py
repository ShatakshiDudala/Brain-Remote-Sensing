#"""
#Inference utilities for brain-inspired remote sensing models
#Handles model inference, metrics calculation, and result processing
#"""

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, jaccard_score
)
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2

def run_inference(model, images, model_type='cnn', batch_size=32, 
                 return_attention=False, return_energy=False):
    """
    Run inference with any model type
    
    Args:
        model: Trained model instance
        images: Input images
        model_type: Type of model ('cnn', 'snn', 'transformer')
        batch_size: Batch size for inference
        return_attention: Whether to return attention maps
        return_energy: Whether to calculate energy consumption
    
    Returns:
        Dictionary with predictions and optional metrics
    """
    start_time = time.time()
    
    # Prepare results dictionary
    results = {
        'predictions': None,
        'probabilities': None,
        'inference_time': 0,
        'energy_consumption': None,
        'attention_maps': None,
        'model_type': model_type
    }
    
    try:
        if model_type.lower() == 'cnn':
            # Standard CNN inference
            predictions, inference_time = model.predict(images, batch_size=batch_size)
            
            # Convert probabilities to class predictions
            class_predictions = np.argmax(predictions, axis=1)
            
            results.update({
                'predictions': class_predictions,
                'probabilities': predictions,
                'inference_time': inference_time
            })
            
            # Energy estimation for CNN
            if return_energy:
                energy = model.get_energy_consumption(len(images))
                results['energy_consumption'] = energy
        
        elif model_type.lower() == 'snn':
            # Spiking Neural Network inference
            predictions, inference_time, energy = model.predict(
                images, batch_size=batch_size
            )
            
            class_predictions = np.argmax(predictions, axis=1)
            
            results.update({
                'predictions': class_predictions,
                'probabilities': predictions,
                'inference_time': inference_time,
                'energy_consumption': energy
            })
            
            # Get spike patterns if available
            if hasattr(model, 'visualize_spikes'):
                spike_data = model.visualize_spikes(images[0])
                results['spike_patterns'] = spike_data
        
        elif model_type.lower() == 'transformer':
            # Vision Transformer inference
            if return_attention:
                predictions, attention_weights, inference_time = model.predict(
                    images, batch_size=batch_size, return_attention=True
                )
                results['attention_maps'] = attention_weights
            else:
                predictions, inference_time = model.predict(
                    images, batch_size=batch_size
                )
            
            class_predictions = np.argmax(predictions, axis=1)
            
            results.update({
                'predictions': class_predictions,
                'probabilities': predictions,
                'inference_time': inference_time
            })
            
            # Energy estimation for Transformer
            if return_energy:
                # Simplified energy estimation
                params = model.model.count_params()
                energy = params * 0.15 * len(images) / 1000  # Rough estimate
                results['energy_consumption'] = energy
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    except Exception as e:
        print(f"Inference error: {str(e)}")
        results['error'] = str(e)
    
    return results

def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None, 
                     segmentation=False):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_names: Names of classes
        segmentation: Whether this is a segmentation task
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # IoU for segmentation tasks
    if segmentation:
        iou_per_class = []
        for i in range(len(np.unique(y_true))):
            intersection = np.sum((y_true == i) & (y_pred == i))
            union = np.sum((y_true == i) | (y_pred == i))
            iou = intersection / (union + 1e-8)
            iou_per_class.append(iou)
        
        metrics['iou_per_class'] = np.array(iou_per_class)
        metrics['mean_iou'] = np.mean(iou_per_class)
    else:
        # Jaccard score for classification
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='weighted')
    
    # Probabilistic metrics (if probabilities available)
    if y_prob is not None:
        try:
            # ROC AUC (for binary/multiclass)
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            
            # Average Precision
            if len(np.unique(y_true)) == 2:
                metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                # For multiclass, calculate per-class and average
                ap_scores = []
                for i in range(y_prob.shape[1]):
                    y_true_binary = (y_true == i).astype(int)
                    ap = average_precision_score(y_true_binary, y_prob[:, i])
                    ap_scores.append(ap)
                metrics['avg_precision'] = np.mean(ap_scores)
        except Exception as e:
            print(f"Error calculating probabilistic metrics: {e}")
    
    # Classification report
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['classification_report'] = report
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
    
    return metrics

def compare_model_performance(results_dict, metrics_to_compare=None):
    """
    Compare performance across multiple models
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        metrics_to_compare: List of metrics to compare
    
    Returns:
        Comparison dataframe and statistics
    """
    import pandas as pd
    
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'f1_score', 'precision', 'recall', 
                            'inference_time', 'energy_consumption']
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            row = {'model': model_name}
            
            for metric in metrics_to_compare:
                if metric in results['metrics']:
                    row[metric] = results['metrics'][metric]
                elif metric in results:
                    row[metric] = results[metric]
                else:
                    row[metric] = None
            
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate statistics
    stats = {}
    for metric in metrics_to_compare:
        if metric in comparison_df.columns:
            values = comparison_df[metric].dropna()
            if len(values) > 0:
                stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'best_model': comparison_df.loc[values.idxmax(), 'model'] if metric != 'inference_time' else comparison_df.loc[values.idxmin(), 'model']
                }
    
    return comparison_df, stats

def post_process_predictions(predictions, probabilities=None, 
                           confidence_threshold=0.5, apply_smoothing=True):
    """
    Post-process model predictions
    
    Args:
        predictions: Raw model predictions
        probabilities: Prediction probabilities
        confidence_threshold: Minimum confidence threshold
        apply_smoothing: Whether to apply spatial smoothing
    
    Returns:
        Processed predictions
    """
    processed_predictions = predictions.copy()
    
    # Apply confidence thresholding
    if probabilities is not None:
        max_probs = np.max(probabilities, axis=-1)
        low_confidence_mask = max_probs < confidence_threshold
        
        # Set low confidence predictions to "uncertain" class or background
        if len(processed_predictions.shape) == 1:
            processed_predictions[low_confidence_mask] = -1  # Uncertain class
        else:
            processed_predictions[low_confidence_mask] = 0  # Background class
    
    # Apply spatial smoothing for segmentation maps
    if apply_smoothing and len(processed_predictions.shape) == 2:
        # Median filter for noise reduction
        processed_predictions = cv2.medianBlur(
            processed_predictions.astype(np.uint8), 5
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_predictions = cv2.morphologyEx(
            processed_predictions, cv2.MORPH_CLOSE, kernel
        )
        processed_predictions = cv2.morphologyEx(
            processed_predictions, cv2.MORPH_OPEN, kernel
        )
    
    return processed_predictions

def ensemble_predictions(prediction_list, method='voting', weights=None):
    """
    Ensemble multiple model predictions
    
    Args:
        prediction_list: List of prediction arrays from different models
        method: Ensembling method ('voting', 'averaging', 'weighted')
        weights: Weights for weighted averaging
    
    Returns:
        Ensembled predictions
    """
    if not prediction_list:
        return None
    
    # Convert to numpy arrays
    predictions = [np.array(pred) for pred in prediction_list]
    
    if method == 'voting':
        # Majority voting
        from scipy import stats
        ensembled = stats.mode(np.stack(predictions), axis=0)[0].squeeze()
    
    elif method == 'averaging':
        # Simple averaging (for probabilities)
        ensembled = np.mean(predictions, axis=0)
        if len(ensembled.shape) > 1:  # If probabilities
            ensembled = np.argmax(ensembled, axis=-1)
    
    elif method == 'weighted':
        # Weighted averaging
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        
        weighted_sum = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_sum += pred * weight
        
        ensembled = weighted_sum
        if len(ensembled.shape) > 1:  # If probabilities
            ensembled = np.argmax(ensembled, axis=-1)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensembled

def calculate_uncertainty(probabilities, method='entropy'):
    """
    Calculate prediction uncertainty
    
    Args:
        probabilities: Model prediction probabilities
        method: Uncertainty calculation method
    
    Returns:
        Uncertainty scores
    """
    if method == 'entropy':
        # Shannon entropy
        uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=-1)
    
    elif method == 'max_prob':
        # 1 - maximum probability
        uncertainty = 1 - np.max(probabilities, axis=-1)
    
    elif method == 'margin':
        # Margin between top two predictions
        sorted_probs = np.sort(probabilities, axis=-1)
        uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return uncertainty

def active_learning_selection(images, model, selection_method='uncertainty', 
                            n_samples=10, diversity_weight=0.1):
    """
    Select samples for active learning
    
    Args:
        images: Pool of unlabeled images
        model: Trained model
        selection_method: Sample selection strategy
        n_samples: Number of samples to select
        diversity_weight: Weight for diversity in selection
    
    Returns:
        Indices of selected samples
    """
    # Get model predictions
    probabilities = model.predict(images)
    
    if selection_method == 'uncertainty':
        # Select most uncertain samples
        uncertainty_scores = calculate_uncertainty(probabilities, method='entropy')
        selected_indices = np.argsort(uncertainty_scores)[-n_samples:]
    
    elif selection_method == 'margin':
        # Select samples with smallest margin
        margin_scores = calculate_uncertainty(probabilities, method='margin')
        selected_indices = np.argsort(margin_scores)[-n_samples:]
    
    elif selection_method == 'diverse_uncertainty':
        # Combine uncertainty with diversity
        uncertainty_scores = calculate_uncertainty(probabilities, method='entropy')
        
        # Simple diversity measure (could be improved with more sophisticated methods)
        selected_indices = []
        remaining_indices = list(range(len(images)))
        
        for _ in range(n_samples):
            if not remaining_indices:
                break
            
            # Calculate combined score
            scores = []
            for idx in remaining_indices:
                uncertainty = uncertainty_scores[idx]
                
                # Diversity score (average distance to already selected)
                if selected_indices:
                    diversity = np.mean([
                        np.linalg.norm(images[idx].flatten() - images[sel_idx].flatten())
                        for sel_idx in selected_indices
                    ])
                else:
                    diversity = 1.0
                
                combined_score = uncertainty + diversity_weight * diversity
                scores.append((combined_score, idx))
            
            # Select best sample
            best_idx = max(scores, key=lambda x: x[0])[1]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    else:
        # Random selection
        selected_indices = np.random.choice(len(images), n_samples, replace=False)
    
    return selected_indices

def benchmark_model_performance(model, test_data, test_labels, 
                              num_runs=10, batch_sizes=[1, 8, 16, 32]):
    """
    Comprehensive model performance benchmarking
    
    Args:
        model: Model to benchmark
        test_data: Test dataset
        test_labels: Test labels
        num_runs: Number of benchmark runs
        batch_sizes: Different batch sizes to test
    
    Returns:
        Benchmark results
    """
    benchmark_results = {
        'accuracy': [],
        'inference_times': [],
        'memory_usage': [],
        'batch_size_performance': {}
    }
    
    # Accuracy benchmark
    for run in range(num_runs):
        predictions = model.predict(test_data)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(test_labels, predictions)
        benchmark_results['accuracy'].append(accuracy)
    
    # Speed benchmark for different batch sizes
    for batch_size in batch_sizes:
        times = []
        
        for run in range(num_runs):
            start_time = time.time()
            model.predict(test_data[:batch_size])
            inference_time = time.time() - start_time
            times.append(inference_time / batch_size)  # Per sample time
        
        benchmark_results['batch_size_performance'][batch_size] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'throughput': batch_size / np.mean(times)  # Samples per second
        }
    
    # Overall statistics
    benchmark_results['accuracy_stats'] = {
        'mean': np.mean(benchmark_results['accuracy']),
        'std': np.std(benchmark_results['accuracy']),
        'min': np.min(benchmark_results['accuracy']),
        'max': np.max(benchmark_results['accuracy'])
    }
    
    return benchmark_results


# Example usage and testing
if __name__ == "__main__":
    print("Inference Utilities - Example Usage")
    
    # Create mock data for testing
    print("\n1. Creating mock data...")
    num_samples = 100
    num_classes = 5
    
    # Mock predictions and ground truth
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_prob = np.random.rand(num_samples, num_classes)
    y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)  # Normalize
    
    print(f"Mock data created: {num_samples} samples, {num_classes} classes")
    
    # Test metrics calculation
    print("\n2. Testing metrics calculation...")
    class_names = ['Water', 'Vegetation', 'Urban', 'Bare Soil', 'Agriculture']
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)
    
    print("Calculated metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Test ensemble predictions
    print("\n3. Testing ensemble predictions...")
    pred1 = np.random.randint(0, num_classes, 50)
    pred2 = np.random.randint(0, num_classes, 50)
    pred3 = np.random.randint(0, num_classes, 50)
    
    ensemble_result = ensemble_predictions([pred1, pred2, pred3], method='voting')
    print(f"Ensemble result shape: {ensemble_result.shape}")
    
    # Test uncertainty calculation
    print("\n4. Testing uncertainty calculation...")
    uncertainty = calculate_uncertainty(y_prob[:10], method='entropy')
    print(f"Uncertainty scores (first 10): {uncertainty}")
    print(f"Mean uncertainty: {np.mean(uncertainty):.3f}")
    
    # Test post-processing
    print("\n5. Testing post-processing...")
    
    # Create mock segmentation map
    seg_map = np.random.randint(0, num_classes, (64, 64))
    seg_probs = np.random.rand(64, 64, num_classes)
    seg_probs = seg_probs / np.sum(seg_probs, axis=-1, keepdims=True)
    
    processed_seg = post_process_predictions(
        seg_map, 
        probabilities=np.max(seg_probs, axis=-1),
        confidence_threshold=0.7,
        apply_smoothing=True
    )
    
    print(f"Original segmentation shape: {seg_map.shape}")
    print(f"Processed segmentation shape: {processed_seg.shape}")
    print(f"Unique classes before: {len(np.unique(seg_map))}")
    print(f"Unique classes after: {len(np.unique(processed_seg))}")
    
    # Test model comparison
    print("\n6. Testing model comparison...")
    
    # Mock results from different models
    mock_results = {
        'CNN': {
            'metrics': {
                'accuracy': 0.912,
                'f1_score': 0.895,
                'precision': 0.903,
                'recall': 0.887
            },
            'inference_time': 0.045,
            'energy_consumption': 156.2
        },
        'SNN': {
            'metrics': {
                'accuracy': 0.932,
                'f1_score': 0.918,
                'precision': 0.925,
                'recall': 0.911
            },
            'inference_time': 0.120,
            'energy_consumption': 23.4
        },
        'Transformer': {
            'metrics': {
                'accuracy': 0.948,
                'f1_score': 0.941,
                'precision': 0.945,
                'recall': 0.937
            },
            'inference_time': 0.089,
            'energy_consumption': 145.7
        }
    }
    
    comparison_df, stats = compare_model_performance(mock_results)
    
    print("Model comparison results:")
    print(comparison_df)
    
    print("\nPerformance statistics:")
    for metric, stat_dict in stats.items():
        print(f"  {metric}:")
        print(f"    Best model: {stat_dict['best_model']}")
        print(f"    Mean: {stat_dict['mean']:.3f}")
        print(f"    Std: {stat_dict['std']:.3f}")
    
    print("\nInference utilities ready!")
    print("Key features:")
    print("- Multi-model inference support")
    print("- Comprehensive metrics calculation")
    print("- Model performance comparison")
    print("- Prediction post-processing")
    print("- Ensemble methods")
    print("- Uncertainty quantification")
    print("- Active learning support")
    print("- Performance benchmarking")