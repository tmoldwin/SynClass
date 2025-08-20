"""
Utility functions for synapse classification
"""
import os
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu=True):
    """Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device: The device to use
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def compute_class_weights(train_files, synapse_map=None):
    """Compute balanced class weights for training.
    
    Args:
        train_files: List of training files
        synapse_map: Mapping from synapse ID to type (if None, will extract from files)
        
    Returns:
        numpy.ndarray: Class weights for balanced training
    """
    if synapse_map is None:
        from data_loader import load_synapse_metadata
        synapse_map = load_synapse_metadata()
    
    # Extract labels (0 for E, 1 for I)
    labels = [1 if synapse_map[int(f.split('_')[0])] == 'I' else 0 for f in train_files]
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.array([0, 1]), 
        y=labels
    )
    
    return class_weights


def count_model_parameters(model):
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def calculate_model_size(model):
    """Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def compute_metrics(y_true, y_pred, class_names=['E', 'I']):
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        
    Returns:
        dict: Dictionary containing various metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Overall metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': {class_names[i]: precision[i] for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: recall[i] for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: f1[i] for i in range(len(class_names))},
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
    
    return metrics


def print_metrics(metrics, logger=None):
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics
        logger: Logger instance (optional)
    """
    def log_print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log_print("\n" + "="*50)
    log_print("CLASSIFICATION METRICS")
    log_print("="*50)
    
    log_print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    
    log_print("\nPer-Class Metrics:")
    log_print("-" * 30)
    for class_name in metrics['precision_per_class'].keys():
        precision = metrics['precision_per_class'][class_name]
        recall = metrics['recall_per_class'][class_name]
        f1 = metrics['f1_per_class'][class_name]
        log_print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    log_print("\nMacro-averaged Metrics:")
    log_print("-" * 25)
    log_print(f"Precision: {metrics['precision_macro']:.4f}")
    log_print(f"Recall:    {metrics['recall_macro']:.4f}")
    log_print(f"F1-Score:  {metrics['f1_macro']:.4f}")
    
    log_print("\nWeighted-averaged Metrics:")
    log_print("-" * 28)
    log_print(f"Precision: {metrics['precision_weighted']:.4f}")
    log_print(f"Recall:    {metrics['recall_weighted']:.4f}")
    log_print(f"F1-Score:  {metrics['f1_weighted']:.4f}")
    
    log_print("="*50)


def ensure_directory_exists(path):
    """Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_model_summary(model, input_size=None):
    """Get a summary of model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size tuple for the model (optional)
        
    Returns:
        dict: Model summary information
    """
    total_params, trainable_params = count_model_parameters(model)
    model_size = calculate_model_size(model)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size,
        'architecture': str(model)
    }
    
    return summary


def print_model_summary(model, input_size=None, logger=None):
    """Print model summary in a formatted way.
    
    Args:
        model: PyTorch model
        input_size: Input size tuple for the model (optional)
        logger: Logger instance (optional)
    """
    def log_print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    summary = get_model_summary(model, input_size)
    
    log_print("\n" + "="*50)
    log_print("MODEL SUMMARY")
    log_print("="*50)
    
    log_print(f"Total Parameters:     {summary['total_parameters']:,}")
    log_print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    log_print(f"Model Size:           {summary['model_size_mb']:.2f} MB")
    
    if input_size:
        log_print(f"Input Size:           {input_size}")
    
    log_print("="*50)


def save_checkpoint(model, optimizer, epoch, loss, path, metadata=None):
    """Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load on (optional)
        
    Returns:
        dict: Checkpoint information
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def format_time(seconds):
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def log_system_info(logger=None):
    """Log system information.
    
    Args:
        logger: Logger instance (optional)
    """
    def log_print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log_print("\n" + "="*50)
    log_print("SYSTEM INFORMATION")
    log_print("="*50)
    
    log_print(f"PyTorch Version: {torch.__version__}")
    log_print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log_print(f"CUDA Version: {torch.version.cuda}")
        log_print(f"GPU Device: {torch.cuda.get_device_name()}")
        log_print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        log_print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    log_print("="*50)
