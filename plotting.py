"""
Plotting utilities for synapse classification
Following user preferences: E=red, I=blue, distinct colors for train/val
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


# Color scheme following user preferences
COLORS = {
    'E': '#FF4444',      # Red for excitatory
    'I': '#4444FF',      # Blue for inhibitory  
    'train': '#2E8B57',  # Sea green for training
    'val': '#FF8C00',    # Dark orange for validation
    'loss': '#8B0000',   # Dark red for loss
    'accuracy': '#006400' # Dark green for accuracy
}


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, 
                         save_path=None, title="Training Progress", show=True):
    """Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses  
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        save_path: Path to save the plot (optional)
        title: Title for the plot
        show: Whether to display the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    ax1.plot(epochs, train_losses, color=COLORS['train'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, color=COLORS['val'], label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, color=COLORS['train'], label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, color=COLORS['val'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=['E', 'I'], save_path=None, 
                         title="Confusion Matrix", show=True, normalize=False):
    """Plot confusion matrix with proper colors.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot (optional)
        title: Title for the plot
        show: Whether to display the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    
    # Create custom colormap with E/I colors
    colors = ['white', COLORS['E'], COLORS['I']]
    n_bins = 100
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    return plt.gcf()


def plot_class_distribution(train_dist, val_dist, save_path=None, title="Class Distribution", show=True):
    """Plot class distribution for train and validation sets.
    
    Args:
        train_dist: Dictionary with training class counts
        val_dist: Dictionary with validation class counts
        save_path: Path to save the plot (optional)
        title: Title for the plot
        show: Whether to display the plot
    """
    classes = ['E', 'I']
    train_counts = [train_dist.get(c, 0) for c in classes]
    val_counts = [val_dist.get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training', 
                   color=COLORS['train'], alpha=0.8)
    bars2 = ax.bar(x + width/2, val_counts, width, label='Validation', 
                   color=COLORS['val'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Synapse Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add class-specific colors as background
    for i, class_name in enumerate(classes):
        ax.axvspan(i-0.4, i+0.4, alpha=0.1, color=COLORS[class_name])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_learning_curves_comparison(results_dict, save_path=None, title="Model Comparison", show=True):
    """Plot learning curves for multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to training results
        save_path: Path to save the plot (optional)
        title: Title for the plot
        show: Whether to display the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        epochs = range(1, len(results['train_losses']) + 1)
        color = colors[i]
        
        # Training loss
        ax1.plot(epochs, results['train_losses'], color=color, 
                label=f'{model_name}', linewidth=2, alpha=0.8)
        
        # Validation loss
        ax2.plot(epochs, results['val_losses'], color=color, 
                label=f'{model_name}', linewidth=2, alpha=0.8)
        
        # Training accuracy
        ax3.plot(epochs, results['train_accuracies'], color=color, 
                label=f'{model_name}', linewidth=2, alpha=0.8)
        
        # Validation accuracy
        ax4.plot(epochs, results['val_accuracies'], color=color, 
                label=f'{model_name}', linewidth=2, alpha=0.8)
    
    # Configure subplots
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_prediction_confidence(predictions, confidences, true_labels, save_path=None, 
                              title="Prediction Confidence Distribution", show=True):
    """Plot distribution of prediction confidences by class.
    
    Args:
        predictions: Predicted labels
        confidences: Prediction confidences
        true_labels: True labels
        save_path: Path to save the plot (optional)
        title: Title for the plot
        show: Whether to display the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)
    
    # Correct vs incorrect predictions
    correct_mask = (predictions == true_labels)
    
    # Plot confidence for correct E predictions
    correct_E = confidences[(predictions == 0) & correct_mask]
    ax1.hist(correct_E, bins=20, color=COLORS['E'], alpha=0.7, edgecolor='black')
    ax1.set_title('Confidence: Correct E Predictions', fontweight='bold')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Plot confidence for correct I predictions
    correct_I = confidences[(predictions == 1) & correct_mask]
    ax2.hist(correct_I, bins=20, color=COLORS['I'], alpha=0.7, edgecolor='black')
    ax2.set_title('Confidence: Correct I Predictions', fontweight='bold')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Plot confidence for incorrect predictions
    incorrect = confidences[~correct_mask]
    ax3.hist(incorrect, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax3.set_title('Confidence: Incorrect Predictions', fontweight='bold')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # Overall confidence distribution
    ax4.hist(confidences, bins=30, color='gray', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidences):.3f}')
    ax4.set_title('Overall Confidence Distribution', fontweight='bold')
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def save_all_plots(training_results, save_dir='figures', model_name='model'):
    """Save all plots for a training run.
    
    Args:
        training_results: Dictionary with training results
        save_dir: Directory to save plots
        model_name: Name of the model for file naming
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Training curves
    plot_training_curves(
        training_results['train_losses'],
        training_results['val_losses'], 
        training_results['train_accuracies'],
        training_results['val_accuracies'],
        save_path=os.path.join(save_dir, f'{model_name}_training_curves.png'),
        title=f'{model_name.title()} Training Progress',
        show=False
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        training_results['final_targets'],
        training_results['final_predictions'],
        save_path=os.path.join(save_dir, f'{model_name}_confusion_matrix.png'),
        title=f'{model_name.title()} Confusion Matrix',
        show=False
    )
    
    print(f"All plots saved to {save_dir}/ directory")
