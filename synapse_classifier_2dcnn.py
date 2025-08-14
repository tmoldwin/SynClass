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
BATCH_SIZE = 16           # Smaller batch for stability
INPUT_XY = 256            # Larger input for better features
EPOCHS = 150              # More epochs for convergence
LR = 5e-6                 # Optimal LR from sweep analysis (fixed)
NUM_WORKERS = 4           # Increased workers for GPU
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
parser.add_argument('--cnn_depth', type=int, default=5, help='Number of convolutional layers (3-7)')
parser.add_argument('--cnn_width', type=int, default=64, help='Base width multiplier for channels')
args = parser.parse_args()
EPOCHS = args.epochs
CNN_DEPTH = args.cnn_depth
CNN_WIDTH = args.cnn_width

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
    
    # Optimize batch size based on GPU memory
    if gpu_memory >= 24:  # High-end GPU (RTX 4090, A100, etc.)
        BATCH_SIZE = 64
        print(f'High-end GPU detected, using batch size: {BATCH_SIZE}')
    elif gpu_memory >= 12:  # Mid-range GPU (RTX 3080, etc.)
        BATCH_SIZE = 32
        print(f'Mid-range GPU detected, using batch size: {BATCH_SIZE}')
    elif gpu_memory >= 8:  # Lower-end GPU
        BATCH_SIZE = 16
        print(f'Lower-end GPU detected, using batch size: {BATCH_SIZE}')
    else:  # Very limited GPU memory
        BATCH_SIZE = 8
        print(f'Limited GPU memory, using batch size: {BATCH_SIZE}')
    
    # Optimize number of workers based on GPU
    if 'A100' in gpu_name or 'H100' in gpu_name:
        NUM_WORKERS = 8
        print(f'High-end GPU detected, using {NUM_WORKERS} workers')
    elif 'RTX' in gpu_name or 'V100' in gpu_name:
        NUM_WORKERS = 6
        print(f'Mid-range GPU detected, using {NUM_WORKERS} workers')
    else:
        NUM_WORKERS = 4
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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
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
        # Base channels: 3 -> width -> width*2 -> width*4 -> width*8 -> width*16
        channels = [3]  # Start with 3 input channels
        for i in range(cnn_depth):
            channels.append(cnn_width * (2 ** i))
        
        # Create convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(cnn_depth):
            conv = nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(channels[i+1])
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        # x shape: [batch, 3, 256, 256]
        
        # Convolutional layers with ReLU and batch norm
        for i in range(self.cnn_depth):
            x = F.relu(self.bn_layers[i](self.conv_layers[i](x)))
            x = self.pool(x)
        
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Classifier
        return self.classifier(attended_features)

def log_epoch_to_csv(run_name, epoch, train_acc, val_acc, overfitting_gap, cnn_depth, cnn_width, sweep_dir=None):
    """Logs the metrics of a training epoch to a centralized CSV file with file locking."""
    if sweep_dir is None:
        log_file = 'sweep_results.csv'
    else:
        os.makedirs(sweep_dir, exist_ok=True)
        log_file = os.path.join(sweep_dir, 'sweep_results.csv')
    lock_file = log_file + '.lock'
    
    # Retry logic for acquiring the lock
    for _ in range(10): # Try for a second
        try:
            # Attempt to acquire lock by creating a unique directory
            os.mkdir(lock_file)
            
            try:
                header = ['run_name', 'epoch', 'train_acc', 'val_acc', 'overfitting_gap', 'cnn_depth', 'cnn_width', 'timestamp']
                file_exists = os.path.isfile(log_file)
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(header)
                    
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([run_name, epoch, f'{train_acc:.2f}', f'{val_acc:.2f}', f'{overfitting_gap:.2f}', cnn_depth, cnn_width, timestamp])
                
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

def plot_learning_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, e_accs, i_accs, run_name=None, sweep_dir=None):
    """Plot comprehensive learning curves and save to figures directory."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots - 3x2 layout for comprehensive view
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Curves - {run_name}', fontsize=16, fontweight='bold')
    
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
    axes[0, 2].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Class-specific accuracies
    axes[1, 0].plot(epochs, e_accs, 'b-', label='E Accuracy', linewidth=2)
    axes[1, 0].plot(epochs, i_accs, 'r-', label='I Accuracy', linewidth=2)
    axes[1, 0].set_title('Class-Specific Accuracies', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting gap
    overfitting_gaps = [train - val for train, val in zip(train_accs, val_accs)]
    axes[1, 1].plot(epochs, overfitting_gaps, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Gap (Train - Val)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final metrics summary
    axes[1, 2].axis('off')
    summary_text = f"""
    Final Results:
    
    Best Val Accuracy: {max(val_accs):.2f}%
    Final Val Accuracy: {val_accs[-1]:.2f}%
    Best Train Accuracy: {max(train_accs):.2f}%
    Final Train Accuracy: {train_accs[-1]:.2f}%
    
    Best E Accuracy: {max(e_accs):.2f}%
    Best I Accuracy: {max(i_accs):.2f}%
    
    Final Overfitting Gap: {overfitting_gaps[-1]:.2f}%
    """
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save to figures directory
    if sweep_dir is None:
        os.makedirs('figures', exist_ok=True)
        save_path = f'figures/2dcnn_training_curves.png'
    else:
        os.makedirs(os.path.join(sweep_dir, 'figures'), exist_ok=True)
        save_path = os.path.join(sweep_dir, 'figures', f'{run_name}_curves.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {save_path}")

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
    
    return train_dataset, test_dataset, synapse_type_map

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, run_name):
    """Train the model with comprehensive logging"""
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    e_accuracies = []
    i_accuracies = []
    
    best_test_acc = 0.0
    best_model_path = MODEL_SAVE_PATHS.get('2dcnn', 'saved_models/best_synapse_model_2dcnn.pth')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Overfitting gap
        overfitting_gap = train_acc - test_acc
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'E Acc: {e_acc:.2f}%, I Acc: {i_acc:.2f}%')
        print(f'Overfitting Gap: {overfitting_gap:.2f}%')
        print(f'Learning Rate: {current_lr:.2e}')
        
        # Log to CSV
        log_epoch_to_csv(run_name, epoch+1, train_acc, test_acc, overfitting_gap, CNN_DEPTH, CNN_WIDTH)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved! Test accuracy: {test_acc:.2f}%')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(test_acc)
        
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
    train_dataset, test_dataset, synapse_type_map = load_and_prepare_data()
    
    if train_dataset is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = CNN2DClassifier(num_classes=2, dropout_rate=DROPOUT_RATE, 
                           cnn_depth=CNN_DEPTH, cnn_width=CNN_WIDTH).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"CNN Depth: {CNN_DEPTH}, CNN Width: {CNN_WIDTH}")
    
    # Loss function and optimizer (using fixed optimal values)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss()
        print("Using Focal Loss (fixed)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        print("Using CrossEntropyLoss with label smoothing (fixed)")
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train model
    print("\nStarting training...")
    train_losses, test_losses, train_accuracies, test_accuracies, learning_rates, e_accuracies, i_accuracies, predictions, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS, args.run_name
    )
    
    # Plot training curves
    plot_learning_curves(train_losses, train_accuracies, test_losses, test_accuracies, 
                        learning_rates, e_accuracies, i_accuracies, args.run_name)
    
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
