import os
import random
import warnings
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from constants import DATA_DIR, CSV_PATH, MODEL_SAVE_PATHS, setup_logging
import datetime
import glob
import csv
import time

warnings.filterwarnings("ignore")

# ------------------------- configuration -------------------------
# Set PyTorch memory optimization
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

BATCH_SIZE = 4            # Further reduced batch size for memory optimization
INPUT_XY = 128            # Keep original input size to preserve image quality
EPOCHS = 150              # More epochs for convergence
LR = 5e-6                 # Optimal LR from sweep analysis (fixed)
NUM_WORKERS = 1           # Further reduced workers for memory optimization
RNG_SEED = 42
DROPOUT_RATE = 0.3        # Optimal dropout from sweep analysis (fixed)
WEIGHT_DECAY = 2e-3       # Optimal weight decay from sweep analysis (fixed)
LABEL_SMOOTHING = 0.1     # Label smoothing instead of high dropout
USE_FOCAL_LOSS = True     # Optimal loss function from sweep analysis (fixed)

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='2D CNN-based synapse classifier')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--run_name', type=str, default=None, help='A unique name for the run for file naming')
parser.add_argument('--cnn_depth', type=int, default=8, help='Number of convolutional layers (8-12)')
parser.add_argument('--cnn_width', type=int, default=128, help='Base width multiplier for channels')
args = parser.parse_args()
EPOCHS = args.epochs
CNN_DEPTH = args.cnn_depth
CNN_WIDTH = args.cnn_width

# Validate depth range
if CNN_DEPTH < 8 or CNN_DEPTH > 12:
    print(f"Warning: CNN depth {CNN_DEPTH} is outside recommended range (8-12)")

# ------------------------- reproducibility ----------------------
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# GPU optimization based on hardware
if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_available = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_free = gpu_memory - gpu_memory_available
        
        print(f'GPU: {gpu_name}')
        print(f'GPU Memory: {gpu_memory:.1f} GB')
        print(f'GPU Memory Available: {gpu_memory_free:.1f} GB')
        
        # Very conservative memory optimization for deep models
        if gpu_memory >= 24:  # High-end GPU (RTX 4090, A100, etc.)
            BATCH_SIZE = 8
            INPUT_XY = 192
            print(f'High-end GPU detected, using batch size: {BATCH_SIZE}, input size: {INPUT_XY}')
        elif gpu_memory >= 16:  # High-mid GPU
            BATCH_SIZE = 6
            INPUT_XY = 160
            print(f'High-mid GPU detected, using batch size: {BATCH_SIZE}, input size: {INPUT_XY}')
        elif gpu_memory >= 12:  # Mid-range GPU (RTX 3080, etc.)
            BATCH_SIZE = 4
            INPUT_XY = 128
            print(f'Mid-range GPU detected, using batch size: {BATCH_SIZE}, input size: {INPUT_XY}')
        elif gpu_memory >= 8:  # Lower-end GPU
            BATCH_SIZE = 2
            INPUT_XY = 96
            print(f'Lower-end GPU detected, using batch size: {BATCH_SIZE}, input size: {INPUT_XY}')
        else:  # Very limited GPU memory
            BATCH_SIZE = 1
            INPUT_XY = 64
            print(f'Limited GPU memory, using batch size: {BATCH_SIZE}, input size: {INPUT_XY}')
        
        # Optimize number of workers based on GPU
        if 'A100' in gpu_name or 'H100' in gpu_name:
            NUM_WORKERS = 4
            print(f'High-end GPU detected, using {NUM_WORKERS} workers')
        elif 'RTX' in gpu_name or 'V100' in gpu_name:
            NUM_WORKERS = 2
            print(f'Mid-range GPU detected, using {NUM_WORKERS} workers')
        else:
            NUM_WORKERS = 1
            print(f'Using {NUM_WORKERS} workers')

# ------------------------- dataset -------------------------
class SynapseDataset2D(Dataset):
    def __init__(self, file_list, synapse_type_map, data_dir, transform=None, is_training=True):
        self.file_list = file_list
        self.synapse_type_map = synapse_type_map
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file = self.file_list[idx]
        try:
            synapse_id = int(file.split('_')[0])
        except (ValueError, IndexError):
            synapse_id = idx
        synapse_type = self.synapse_type_map.get(synapse_id, 'Unknown')
        
        # Load raw data
        file_path = os.path.join(self.data_dir, file)
        raw_data = np.load(file_path)
        
        # Load masks
        pre_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'post_syn_n_mask.npy'))
        pre_mask = np.load(pre_mask_path)
        post_mask = np.load(post_mask_path)
        
        # Create 2D representation by taking the middle slice or max projection
        if raw_data.ndim == 3:
            # Take middle slice for 2D representation
            z_middle = raw_data.shape[2] // 2
            data_slice = raw_data[:, :, z_middle]
            pre_slice = pre_mask[:, :, z_middle]
            post_slice = post_mask[:, :, z_middle]
        else:
            # Already 2D
            data_slice = raw_data
            pre_slice = pre_mask
            post_slice = post_mask
        
        # Resize to fixed size (using cv2 for better interpolation)
        data_slice = cv2.resize(data_slice, (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_AREA)
        pre_slice = cv2.resize(pre_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)
        post_slice = cv2.resize(post_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.is_training:
            data_slice, pre_slice, post_slice = self._augment(data_slice, pre_slice, post_slice)

        # Percentile normalisation
        if data_slice.max() > data_slice.min():
            non_zero_mask = data_slice > 0
            if np.any(non_zero_mask):
                p5, p95 = np.percentile(data_slice[non_zero_mask], [5, 95])
                data_slice = np.clip((data_slice - p5) / (p95 - p5 + 1e-8), 0, 1)
            else: # All zero
                data_slice = np.zeros((INPUT_XY, INPUT_XY))
        
        image = np.stack([data_slice, pre_slice, post_slice], axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        # Label: 0 for E, 1 for I
        label = 1 if synapse_type == 'I' else 0
        
        return image, label, synapse_id
    
    def _augment(self, data_slice, pre_slice, post_slice):
        """Apply data augmentation"""
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            center = (INPUT_XY // 2, INPUT_XY // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            data_slice = cv2.warpAffine(data_slice, M, (INPUT_XY, INPUT_XY))
            pre_slice = cv2.warpAffine(pre_slice, M, (INPUT_XY, INPUT_XY))
            post_slice = cv2.warpAffine(post_slice, M, (INPUT_XY, INPUT_XY))
        
        # Random horizontal flip
        if random.random() > 0.5:
            data_slice = cv2.flip(data_slice, 1)
            pre_slice = cv2.flip(pre_slice, 1)
            post_slice = cv2.flip(post_slice, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            data_slice = cv2.flip(data_slice, 0)
            pre_slice = cv2.flip(pre_slice, 0)
            post_slice = cv2.flip(post_slice, 0)
        
        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-0.1, 0.1)  # Brightness
            data_slice = cv2.convertScaleAbs(data_slice, alpha=alpha, beta=beta)
            data_slice = np.clip(data_slice, 0, 1)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, data_slice.shape)
            data_slice = np.clip(data_slice + noise, 0, 1)
        
        return data_slice, pre_slice, post_slice

# ------------------------- focal loss --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------- 2D CNN MODEL --------------------------
class CNN2DClassifier(nn.Module):
    """2D CNN classifier with attention mechanism and enhanced classifier head"""
    def __init__(self, num_classes=2, dropout_rate=0.3, cnn_depth=5, cnn_width=64):
        super().__init__()
        
        self.cnn_depth = cnn_depth
        self.cnn_width = cnn_width
        
        # Calculate channel progression based on width multiplier
        # For very deep models, cap the maximum channels to prevent memory issues
        channels = [3]  # Start with 3 input channels
        for i in range(cnn_depth):
            # Cap maximum channels to prevent memory overflow
            max_channels = min(cnn_width * (2 ** i), 1024)  # Cap at 1024 channels
            channels.append(max_channels)
        
        # Create convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(cnn_depth):
            conv = nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(channels[i+1])
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
        
        # Calculate expected feature map size after all pooling operations
        # Starting with INPUT_XY x INPUT_XY, each pooling reduces by factor of 2
        expected_size = INPUT_XY // (2 ** cnn_depth)
        
        # Adjust pooling strategy based on expected size
        if expected_size < 1:
            # For very deep models, use adaptive pooling after fewer layers
            self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_every_n_layers = max(1, cnn_depth // 8)  # Pool every N layers instead of every layer
        elif expected_size < 4:
            # For moderately deep models
            self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_every_n_layers = 1  # Pool every layer
        else:
            # For shallow models
            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_every_n_layers = 1  # Pool every layer
        
        # Final feature size is the last channel count
        final_features = channels[-1]
        
        # ATTENTION MECHANISM for better feature focus
        self.attention = nn.Sequential(
            nn.Linear(final_features, final_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(final_features // 4, final_features),
            nn.Sigmoid()
        )
        
        # ENHANCED CLASSIFIER HEAD (similar to ResNet version)
        classifier_layers = []
        
        # First layer: final_features -> final_features//2
        classifier_layers.extend([
            nn.Linear(final_features, final_features // 2),
            nn.BatchNorm1d(final_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        # Second layer: final_features//2 -> final_features//4
        classifier_layers.extend([
            nn.Linear(final_features // 2, final_features // 4),
            nn.BatchNorm1d(final_features // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        # Third layer: final_features//4 -> final_features//8
        classifier_layers.extend([
            nn.Linear(final_features // 4, final_features // 8),
            nn.BatchNorm1d(final_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        ])
        
        # Fourth layer: final_features//8 -> final_features//16
        classifier_layers.extend([
            nn.Linear(final_features // 8, final_features // 16),
            nn.BatchNorm1d(final_features // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25)
        ])
        
        # Final layer: final_features//16 -> num_classes
        classifier_layers.append(nn.Linear(final_features // 16, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        # x shape: [batch, 3, H, W]
        
        # Convolutional layers with ReLU and batch norm
        for i in range(self.cnn_depth):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))
            
            # Apply pooling based on the adaptive strategy
            if (i + 1) % self.pool_every_n_layers == 0:
                x = self.pool(x)
            
            # Safety check: if feature maps become too small, apply adaptive pooling
            if x.shape[2] <= 1 or x.shape[3] <= 1:
                x = self.adaptive_pool(x)
                break
        
        # Final adaptive pooling if not already applied
        if x.shape[2] > 1 or x.shape[3] > 1:
            x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Classifier
        return self.classifier(attended_features)

def log_epoch_to_csv(run_name, epoch, train_acc, val_acc, overfitting_gap, cnn_depth, cnn_width, 
                    train_confusion_matrix, val_confusion_matrix, sweep_dir=None):
    """Logs the metrics of a training epoch to a centralized CSV file with file locking."""
    # Always use sweep directory if available
    if sweep_dir is None:
        sweep_dir = os.getenv('SWEEP_MASTER_DIR', '.')
    
    os.makedirs(sweep_dir, exist_ok=True)
    log_file = os.path.join(sweep_dir, 'sweep_results.csv')
    lock_file = log_file + '.lock'
    
    # Retry logic for acquiring the lock
    for _ in range(10): # Try for a second
        try:
            # Attempt to acquire lock by creating a unique directory
            os.mkdir(lock_file)
            
            try:
                header = ['run_name', 'epoch', 'train_acc', 'val_acc', 'overfitting_gap', 'cnn_depth', 'cnn_width', 
                         'train_e_correct', 'train_e_incorrect', 'train_i_correct', 'train_i_incorrect',
                         'val_e_correct', 'val_e_incorrect', 'val_i_correct', 'val_i_incorrect',
                         'train_e_total', 'train_i_total', 'val_e_total', 'val_i_total', 'timestamp']
                file_exists = os.path.isfile(log_file)
                
                # Extract confusion matrix values
                train_e_correct = train_confusion_matrix[0, 0] if train_confusion_matrix is not None else 0
                train_e_incorrect = train_confusion_matrix[0, 1] if train_confusion_matrix is not None else 0
                train_i_correct = train_confusion_matrix[1, 1] if train_confusion_matrix is not None else 0
                train_i_incorrect = train_confusion_matrix[1, 0] if train_confusion_matrix is not None else 0
                
                val_e_correct = val_confusion_matrix[0, 0] if val_confusion_matrix is not None else 0
                val_e_incorrect = val_confusion_matrix[0, 1] if val_confusion_matrix is not None else 0
                val_i_correct = val_confusion_matrix[1, 1] if val_confusion_matrix is not None else 0
                val_i_incorrect = val_confusion_matrix[1, 0] if val_confusion_matrix is not None else 0
                
                # Calculate totals
                train_e_total = train_e_correct + train_e_incorrect
                train_i_total = train_i_correct + train_i_incorrect
                val_e_total = val_e_correct + val_e_incorrect
                val_i_total = val_i_correct + val_i_incorrect
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(header)
                    
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([run_name, epoch, f'{train_acc:.2f}', f'{val_acc:.2f}', f'{overfitting_gap:.2f}', 
                                   cnn_depth, cnn_width, train_e_correct, train_e_incorrect, train_i_correct, train_i_incorrect,
                                   val_e_correct, val_e_incorrect, val_i_correct, val_i_incorrect,
                                   train_e_total, train_i_total, val_e_total, val_i_total, timestamp])
                
                # Lock released, exit retry loop
                break

            finally:
                # Always release the lock
                os.rmdir(lock_file)
                
        except FileExistsError:
            # Lock is held by another process, wait and retry
            time.sleep(0.1)
    else:
        # If the lock is not acquired after multiple retries, log an error and move on
        print(f"Warning: Could not acquire lock to write to {log_file} for run {run_name}, epoch {epoch}. Skipping log entry.")

def plot_epoch_progress(train_losses, train_accs, val_losses, val_accs, learning_rates, e_accs, i_accs, 
                       run_name=None, sweep_dir=None, cnn_depth=None, cnn_width=None, current_epoch=None,
                       train_confusion_matrix=None, val_confusion_matrix=None):
    """Plot current training progress and save with network size and accuracies in filename."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots - 3x3 layout for comprehensive view including confusion matrices
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # Create title with network info and current accuracies
    current_train_acc = train_accs[-1] if train_accs else 0
    current_val_acc = val_accs[-1] if val_accs else 0
    title = f'Training Progress - {run_name} | Epoch {current_epoch} | Net: {cnn_depth}L-{cnn_width}W | Train: {current_train_acc:.1f}% | Val: {current_val_acc:.1f}%'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate curve
    if learning_rates:
        axes[0, 2].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Class-specific accuracies
    if e_accs and i_accs:
        axes[1, 0].plot(epochs, e_accs, 'b-', label='E Accuracy', linewidth=2)
        axes[1, 0].plot(epochs, i_accs, 'r-', label='I Accuracy', linewidth=2)
        axes[1, 0].set_title('Class-Specific Accuracies', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    if train_accs and val_accs:
        overfitting_gaps = [train - val for train, val in zip(train_accs, val_accs)]
        axes[1, 1].plot(epochs, overfitting_gaps, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Gap (Train - Val)', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Training Confusion Matrix
    if train_confusion_matrix is not None:
        im1 = axes[1, 2].imshow(train_confusion_matrix, interpolation='nearest', cmap=plt.cm.Reds)
        axes[1, 2].set_title('Training Confusion Matrix', fontweight='bold')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['E', 'I'])
        axes[1, 2].set_yticklabels(['E', 'I'])
        
        # Add text annotations
        total_train = train_confusion_matrix.sum()
        for i in range(2):
            for j in range(2):
                count = train_confusion_matrix[i, j]
                percentage = 100. * count / total_train
                axes[1, 2].text(j, i, f'{count}\n({percentage:.1f}%)',
                               ha='center', va='center', fontweight='bold',
                               color='white' if count > total_train/4 else 'black')
    
    # Validation Confusion Matrix
    if val_confusion_matrix is not None:
        im2 = axes[2, 0].imshow(val_confusion_matrix, interpolation='nearest', cmap=plt.cm.Reds)
        axes[2, 0].set_title('Validation Confusion Matrix', fontweight='bold')
        axes[2, 0].set_xlabel('Predicted')
        axes[2, 0].set_ylabel('Actual')
        axes[2, 0].set_xticks([0, 1])
        axes[2, 0].set_yticks([0, 1])
        axes[2, 0].set_xticklabels(['E', 'I'])
        axes[2, 0].set_yticklabels(['E', 'I'])
        
        # Add text annotations
        total_val = val_confusion_matrix.sum()
        for i in range(2):
            for j in range(2):
                count = val_confusion_matrix[i, j]
                percentage = 100. * count / total_val
                axes[2, 0].text(j, i, f'{count}\n({percentage:.1f}%)',
                               ha='center', va='center', fontweight='bold',
                               color='white' if count > total_val/4 else 'black')
    
    # Current metrics summary
    axes[2, 1].axis('off')
    # Calculate best accuracies safely
    best_val_acc = max(val_accs) if val_accs else 0
    best_train_acc = max(train_accs) if train_accs else 0
    
    summary_text = f"""
    Current Progress (Epoch {current_epoch}):
    
    Current Val Accuracy: {current_val_acc:.2f}%
    Current Train Accuracy: {current_train_acc:.2f}%
    Best Val Accuracy: {best_val_acc:.2f}%
    Best Train Accuracy: {best_train_acc:.2f}%
    
    Network: {cnn_depth} layers, {cnn_width} width
    Overfitting Gap: {current_train_acc - current_val_acc:.2f}%
    """
    axes[2, 1].text(0.1, 0.5, summary_text, transform=axes[2, 1].transAxes, 
                   fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    # Sample counts summary
    axes[2, 2].axis('off')
    if train_confusion_matrix is not None and val_confusion_matrix is not None:
        train_e_total = train_confusion_matrix[0, :].sum()
        train_i_total = train_confusion_matrix[1, :].sum()
        val_e_total = val_confusion_matrix[0, :].sum()
        val_i_total = val_confusion_matrix[1, :].sum()
        
        sample_text = f"""
    Sample Counts:
    
    Training Set:
    • E: {train_e_total} samples
    • I: {train_i_total} samples
    • Total: {train_e_total + train_i_total} samples
    
    Validation Set:
    • E: {val_e_total} samples
    • I: {val_i_total} samples
    • Total: {val_e_total + val_i_total} samples
    
    Class Balance:
    • Train: {train_e_total/(train_e_total+train_i_total)*100:.1f}% E, {train_i_total/(train_e_total+train_i_total)*100:.1f}% I
    • Val: {val_e_total/(val_e_total+val_i_total)*100:.1f}% E, {val_i_total/(val_e_total+val_i_total)*100:.1f}% I
    """
        axes[2, 2].text(0.1, 0.5, sample_text, transform=axes[2, 2].transAxes, 
                       fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Create filename with network size and current accuracies
    filename = f"{run_name}_epoch{current_epoch:03d}_net{cnn_depth}L{cnn_width}W_train{current_train_acc:.1f}_val{current_val_acc:.1f}.png"
    
    # Save to figures directory
    if sweep_dir is None:
        os.makedirs('figures', exist_ok=True)
        save_path = f'figures/{filename}'
        figures_dir = 'figures'
    else:
        os.makedirs(os.path.join(sweep_dir, 'figures'), exist_ok=True)
        save_path = os.path.join(sweep_dir, 'figures', filename)
        figures_dir = os.path.join(sweep_dir, 'figures')
    
    # Delete previous epoch figures for this run to save disk space
    if current_epoch > 1:
        import glob
        # Find all previous epoch figures for this run
        pattern = f"{run_name}_epoch*_net{cnn_depth}L{cnn_width}W_*.png"
        previous_files = glob.glob(os.path.join(figures_dir, pattern))
        
        # Delete files from previous epochs (keep only current epoch)
        for old_file in previous_files:
            try:
                os.remove(old_file)
                print(f"Deleted previous epoch figure: {old_file}")
            except OSError as e:
                print(f"Could not delete {old_file}: {e}")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"Progress plot saved: {save_path}")
    
    return save_path

def load_and_prepare_data():
    """Load and prepare the dataset"""
    logger = setup_logging('2dcnn')
    logger.info("Loading synapse data...")
    
    # Load CSV data
    if not os.path.exists(CSV_PATH):
        logger.error(f"Error: {CSV_PATH} not found!")
        return None, None, None
    
    synapse_data = pd.read_csv(CSV_PATH)
    synapse_type_map = {row['id_']: row['pre_clf_type'] for _, row in synapse_data.iterrows()}
    
    # Get all synapse files
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('syn.npy')]
    
    # Filter files that have valid synapse types
    valid_files = []
    for file in all_files:
        try:
            synapse_id = int(file.split('_')[0])
            if synapse_id in synapse_type_map and synapse_type_map[synapse_id] in ['E', 'I']:
                valid_files.append(file)
        except (ValueError, IndexError):
            continue
    
    logger.info(f"Found {len(valid_files)} valid synapse files")
    logger.info(f"E synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'E')}")
    logger.info(f"I synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'I')}")
    
    if len(valid_files) == 0:
        logger.error("No valid synapse files found!")
        return None, None, None
    
    # Split data
    train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=RNG_SEED, 
                                              stratify=[synapse_type_map[int(f.split('_')[0])] for f in valid_files])
    
    # Create datasets
    train_dataset = SynapseDataset2D(train_files, synapse_type_map, DATA_DIR, is_training=True)
    test_dataset = SynapseDataset2D(test_files, synapse_type_map, DATA_DIR, is_training=False)
    
    # Calculate class distribution for imbalance handling
    train_labels = [synapse_type_map[int(f.split('_')[0])] for f in train_files]
    test_labels = [synapse_type_map[int(f.split('_')[0])] for f in test_files]
    
    train_e_count = sum(1 for label in train_labels if label == 'E')
    train_i_count = sum(1 for label in train_labels if label == 'I')
    test_e_count = sum(1 for label in test_labels if label == 'E')
    test_i_count = sum(1 for label in test_labels if label == 'I')
    
    logger.info(f"Training set - E: {train_e_count}, I: {train_i_count}, Ratio: {train_e_count/train_i_count:.2f}:1")
    logger.info(f"Test set - E: {test_e_count}, I: {test_i_count}, Ratio: {test_e_count/test_i_count:.2f}:1")
    
    # Calculate class weights for balanced loss
    total_train = len(train_labels)
    class_weights = torch.tensor([
        total_train / (2 * train_e_count),  # Weight for E class
        total_train / (2 * train_i_count)   # Weight for I class
    ], dtype=torch.float32)
    
    logger.info(f"Class weights: E={class_weights[0]:.3f}, I={class_weights[1]:.3f}")
    
    return train_dataset, test_dataset, synapse_type_map, class_weights

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, run_name):
    """Train the model with comprehensive logging and early stopping"""
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    e_accuracies = []
    i_accuracies = []
    
    best_test_acc = 0.0
    best_model_path = MODEL_SAVE_PATHS.get('2dcnn', 'saved_models/best_synapse_model_2dcnn.pth')
    
    # Early stopping parameters
    patience = 15  # Number of epochs to wait for improvement
    min_delta = 0.1  # Minimum improvement required
    patience_counter = 0
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_predictions = []
        all_train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, labels, _) in enumerate(train_pbar):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'Batch': f'{batch_idx}/{len(train_loader)}'
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        e_correct = 0
        e_total = 0
        i_correct = 0
        i_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for data, labels, _ in test_pbar:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Class-specific accuracy
                for i, label in enumerate(labels):
                    if label == 0:  # E
                        e_total += 1
                        if predicted[i] == 0:
                            e_correct += 1
                    else:  # I
                        i_total += 1
                        if predicted[i] == 1:
                            i_correct += 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*test_correct/test_total:.2f}%'
                })
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Class-specific accuracies
        e_acc = 100. * e_correct / e_total if e_total > 0 else 0
        i_acc = 100. * i_correct / i_total if i_total > 0 else 0
        e_accuracies.append(e_acc)
        i_accuracies.append(i_acc)
        
        # Calculate confusion matrices
        from sklearn.metrics import confusion_matrix
        val_confusion_matrix = confusion_matrix(all_labels, all_predictions)
        train_confusion_matrix = confusion_matrix(all_train_labels, all_train_predictions)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Overfitting gap
        overfitting_gap = train_acc - test_acc
        
        # Print epoch summary with class imbalance info
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'E Acc: {e_acc:.2f}% ({e_correct}/{e_total}), I Acc: {i_acc:.2f}% ({i_correct}/{i_total})')
        print(f'Class Balance: E:I = {e_total}:{i_total} = {e_total/i_total:.2f}:1')
        print(f'Overfitting Gap: {overfitting_gap:.2f}%')
        print(f'Learning Rate: {current_lr:.2e}')
        
        # Log to CSV with confusion matrix data
        log_epoch_to_csv(run_name, epoch+1, train_acc, test_acc, overfitting_gap, CNN_DEPTH, CNN_WIDTH,
                        train_confusion_matrix, val_confusion_matrix)
        
        # Plot progress after every epoch
        sweep_dir = os.getenv('SWEEP_MASTER_DIR', None)
        plot_epoch_progress(train_losses, train_accuracies, test_losses, test_accuracies, 
                           learning_rates, e_accuracies, i_accuracies, run_name, sweep_dir, 
                           CNN_DEPTH, CNN_WIDTH, epoch+1, train_confusion_matrix, val_confusion_matrix)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved! Test accuracy: {test_acc:.2f}%')
        
        # Early stopping logic
        if test_acc > best_val_acc + min_delta:
            best_val_acc = test_acc
            patience_counter = 0
            print(f'Validation accuracy improved to {test_acc:.2f}% - resetting patience counter')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (patience: {patience})')
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
            print(f'No improvement for {patience} epochs')
            break
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(test_acc)
        
        # Memory cleanup for deep models
        if CNN_DEPTH >= 10:
            torch.cuda.empty_cache()
        
        print('-' * 50)
    
    return train_losses, test_losses, train_accuracies, test_accuracies, learning_rates, e_accuracies, i_accuracies, all_predictions, all_labels

def main():
    """Main training function"""
    print("Starting 2D CNN Synapse Classification Training")
    print("=" * 60)
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"2dcnn_d{CNN_DEPTH}_w{CNN_WIDTH}"
    
    print(f"Run name: {args.run_name}")
    print(f"Configuration:")
    print(f"  CNN Depth: {CNN_DEPTH}")
    print(f"  CNN Width: {CNN_WIDTH}")
    print(f"  Learning Rate: {LR} (fixed)")
    print(f"  Dropout Rate: {DROPOUT_RATE} (fixed)")
    print(f"  Weight Decay: {WEIGHT_DECAY} (fixed)")
    print(f"  Use Focal Loss: {USE_FOCAL_LOSS} (fixed)")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Input Size: {INPUT_XY}x{INPUT_XY}")
    
    # Load data
    train_dataset, test_dataset, synapse_type_map, class_weights = load_and_prepare_data()
    
    if train_dataset is None:
        print("Failed to load data. Exiting.")
        return
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    print(f"Class weights (E, I): {class_weights.cpu().numpy()}")
    
    # Create data loaders
    print(f"Creating data loaders with {NUM_WORKERS} workers...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,  # Disable persistent workers to avoid hanging
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,  # Disable persistent workers to avoid hanging
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    print("Data loaders created successfully!")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = CNN2DClassifier(num_classes=2, dropout_rate=DROPOUT_RATE, 
                           cnn_depth=CNN_DEPTH, cnn_width=CNN_WIDTH).to(device)
    
    # Enable gradient checkpointing for memory optimization on deep models
    if CNN_DEPTH >= 10:
        model.use_checkpoint = True
        print("Gradient checkpointing enabled for deep model")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"CNN Depth: {CNN_DEPTH}, CNN Width: {CNN_WIDTH}")
    print("Model initialized successfully!")
    
    # Loss function and optimizer (using fixed optimal values with class weights)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights)
        print("Using Focal Loss with class weights (fixed)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        print("Using CrossEntropyLoss with class weights and label smoothing (fixed)")
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train model
    print("\nStarting training...")
    train_losses, test_losses, train_accuracies, test_accuracies, learning_rates, e_accuracies, i_accuracies, predictions, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS, args.run_name
    )
    
    # Final progress plot (already done every epoch, but this is the final one)
    # For the final plot, we'll use the last epoch's confusion matrices
    sweep_dir = os.getenv('SWEEP_MASTER_DIR', None)
    plot_epoch_progress(train_losses, train_accuracies, test_losses, test_accuracies, 
                        learning_rates, e_accuracies, i_accuracies, args.run_name, sweep_dir, 
                        CNN_DEPTH, CNN_WIDTH, EPOCHS, None, None)  # No confusion matrices for final plot
    
    # Final evaluation
    print("\nFinal Results:")
    print("=" * 40)
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best E Accuracy: {max(e_accuracies):.2f}%")
    print(f"Best I Accuracy: {max(i_accuracies):.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['E', 'I']))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
