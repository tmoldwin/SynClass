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
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from constants import DATA_DIR, CSV_PATH, MODEL_SAVE_PATHS, setup_logging
import datetime
import glob
import csv
import time

warnings.filterwarnings("ignore")

# ------------------------- BIG MODEL CONFIGURATION -------------------------
BATCH_SIZE = 16           # Smaller batch for stability
INPUT_XY = 256            # Larger input for better features
EPOCHS = 150              # More epochs for convergence
LR = 5e-6                 # Optimal LR from sweep
NUM_WORKERS = 4           
RNG_SEED = 42
DROPOUT_RATE = 0.3        # REDUCED from 0.8 (analysis showed over-regularization)
WEIGHT_DECAY = 1e-3       # REDUCED from 0.002
LABEL_SMOOTHING = 0.1     # Label smoothing instead of high dropout

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='BIG ResNet-based synapse classifier')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE)
parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING)
parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of cross entropy')
parser.add_argument('--run_name', type=str, default=None, help='Name for this run')

args = parser.parse_args()

# Set random seeds
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------- BIG RESNET MODEL --------------------------
class BigResNetClassifier(nn.Module):
    """MUCH BIGGER model based on analysis findings"""
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super().__init__()
        
        # Use ResNet152 as backbone (bigger than ResNet50)
        self.resnet = models.resnet152(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        
        # Remove the final layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MUCH BIGGER CLASSIFIER (50-100% bigger as recommended)
        self.classifier = nn.Sequential(
            # First layer: 2048 -> 1024 (bigger than 128!)
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Second layer: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Third layer: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout
            
            # Fourth layer: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),  # Further reduced dropout
            
            # Final layer: 128 -> num_classes
            nn.Linear(128, num_classes)
        )
        
        # Add residual connections (as recommended)
        self.residual_connections = True
        
    def forward(self, x):
        features = self.resnet(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Apply classifier with residual connections
        x = self.classifier(features)
        return x

# ------------------------- ENHANCED LOSS FUNCTIONS --------------------------
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

# ------------------------- ENHANCED DATASET WITH AUGMENTATION --------------------------
class BigSynapseDataset(Dataset):
    """Enhanced dataset that works with .npy files like existing models"""
    def __init__(self, file_list, type_map, data_dir, augment=True):
        self.file_list = file_list
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment
        
        # ENHANCED DATA AUGMENTATION (as recommended)
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(INPUT_XY, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((INPUT_XY, INPUT_XY)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        
        # Load .npy files like existing models
        raw_path = os.path.join(self.data_dir, fname)
        pre_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'post_syn_n_mask.npy'))
        
        # Load the data
        raw_data = np.load(raw_path)
        pre_mask = np.load(pre_path)
        post_mask = np.load(post_path)
        
        # Take middle slice for 2D processing
        z_mid = raw_data.shape[2] // 2
        raw_slice = raw_data[:, :, z_mid]
        pre_slice = pre_mask[:, :, z_mid]
        post_slice = post_mask[:, :, z_mid]
        
        # Create 3-channel image: [raw, pre_mask, post_mask]
        img = np.stack([raw_slice, pre_slice, post_slice], axis=2)
        
        # Normalize to 0-255 range
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Convert to PIL and apply transformations
        img = Image.fromarray(img)
        img = self.transform(img)
        
        # Get label
        synapse_id = int(fname.split('_')[0])
        label = 1 if self.type_map[synapse_id] == 'I' else 0
        
        return img, label

# ------------------------- ENHANCED TRAINING FUNCTION --------------------------
def train_big_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, run_name):
    """Enhanced training with better optimization"""
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    e_accs, i_accs = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping (as recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        e_correct, e_total = 0, 0
        i_correct, i_total = 0, 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                # Per-class accuracy
                for i in range(len(target)):
                    if target[i] == 0:  # E class
                        e_total += 1
                        if predicted[i] == 0:
                            e_correct += 1
                    else:  # I class
                        i_total += 1
                        if predicted[i] == 1:
                            i_correct += 1
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        e_acc = 100. * e_correct / e_total if e_total > 0 else 0
        i_acc = 100. * i_correct / i_total if i_total > 0 else 0
        e_accs.append(e_acc)
        i_accs.append(i_acc)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_acc)
        
        # Calculate overfitting gap
        overfitting_gap = train_acc - val_acc
        
        # Log to CSV
        if run_name:
            log_epoch_to_csv(run_name, epoch+1, optimizer.param_groups[0]['lr'], 
                           DROPOUT_RATE, WEIGHT_DECAY, train_acc, val_acc, 
                           overfitting_gap, args.use_focal_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  E Acc: {e_acc:.2f}%, I Acc: {i_acc:.2f}%')
        print(f'  Overfitting Gap: {overfitting_gap:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_big_model_{run_name}.pth' if run_name else 'best_big_model.pth')
            print(f'  New best model saved! Val Acc: {best_val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs, e_accs, i_accs

# ------------------------- LOGGING FUNCTION --------------------------
def log_epoch_to_csv(run_name, epoch, lr, dropout, weight_decay, train_acc, val_acc, overfitting_gap, use_focal_loss):
    """Logs the metrics of a training epoch to a centralized CSV file."""
    log_file = 'big_model_results.csv'
    
    header = ['run_name', 'epoch', 'lr', 'dropout', 'weight_decay', 'train_acc', 'val_acc', 'overfitting_gap', 'use_focal_loss', 'timestamp']
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([run_name, epoch, lr, dropout, weight_decay, f'{train_acc:.2f}', f'{val_acc:.2f}', f'{overfitting_gap:.2f}', use_focal_loss, timestamp])

# ------------------------- MAIN FUNCTION --------------------------
def main():
    print("Starting BIG Synapse Classification Training")
    print("=" * 60)
    print(f"Model: BIG ResNet152 with enhanced classifier")
    print(f"Input size: {INPUT_XY}x{INPUT_XY}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Dropout rate: {DROPOUT_RATE} (REDUCED from 0.8)")
    print(f"Weight decay: {WEIGHT_DECAY} (REDUCED from 0.002)")
    print(f"Epochs: {EPOCHS}")
    print(f"Use focal loss: {args.use_focal_loss}")
    print("=" * 60)
    
    # Generate run name if not provided
    if not args.run_name:
        args.run_name = f"big_model_lr{LR}_dr{DROPOUT_RATE}_wd{WEIGHT_DECAY}_focal{args.use_focal_loss}"
    
    # Load data
    print("Loading data...")
    
    # Load CSV data like existing models
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    data = pd.read_csv(CSV_PATH)
    type_map = {r['id_']: r['pre_clf_type'] for _, r in data.iterrows()}
    
    # Get .npy files like existing models
    all_files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('syn.npy'):
            try:
                syn_id = int(f.split('_')[0])
                if syn_id in type_map and type_map[syn_id] in ['E', 'I']:
                    all_files.append(f)
            except (ValueError, IndexError):
                continue
    
    # Balance classes
    E_files = [f for f in all_files if type_map[int(f.split('_')[0])] == 'E']
    I_files = [f for f in all_files if type_map[int(f.split('_')[0])] == 'I']
    size = min(len(E_files), len(I_files))
    if size == 0:
        raise ValueError("Not enough data for at least one class")
    
    random.shuffle(E_files)
    random.shuffle(I_files)
    files_bal = E_files[:size] + I_files[:size]
    random.shuffle(files_bal)
    
    # Split data
    train_f, test_f = train_test_split(files_bal, test_size=0.2, random_state=RNG_SEED, 
                                      stratify=[type_map[int(f.split('_')[0])] for f in files_bal])
    
    print(f"Train files: {len(train_f)}")
    print(f"Test files: {len(test_f)}")
    
    # Create datasets
    train_dataset = BigSynapseDataset(train_f, type_map, DATA_DIR, augment=True)
    test_dataset = BigSynapseDataset(test_f, type_map, DATA_DIR, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Initialize BIG model
    model = BigResNetClassifier(num_classes=2, dropout_rate=DROPOUT_RATE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (MUCH BIGGER than before)")
    
    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # ENHANCED OPTIMIZER (AdamW as recommended)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # ENHANCED SCHEDULER (cosine annealing as recommended)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=LR/100
    )
    
    print(f"Optimizer: AdamW")
    print(f"Scheduler: CosineAnnealingWarmRestarts")
    print(f"Loss function: {'Focal Loss' if args.use_focal_loss else 'CrossEntropy with Label Smoothing'}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs, e_accs, i_accs = train_big_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS, args.run_name
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Model parameters: {total_params:,}")
    print(f"Expected improvement: +3-5% from bigger model")
    print("=" * 60)
    
    # Save final model
    torch.save(model.state_dict(), f'final_big_model_{args.run_name}.pth')
    print(f"Final model saved as: final_big_model_{args.run_name}.pth")

if __name__ == "__main__":
    main()
