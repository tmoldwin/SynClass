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
import matplotlib.pyplot as plt
import seaborn as sns
from constants import DATA_DIR, CSV_PATH, MODEL_SAVE_PATHS, setup_logging
import datetime

warnings.filterwarnings("ignore")

# ------------------------- configuration -------------------------
BATCH_SIZE = 32           # Larger batch size for better GPU utilization
INPUT_XY = 224            # Standard ResNet input size
EPOCHS = 100              # More epochs for better convergence
LR = 5e-6                 # Even lower LR to prevent overfitting
NUM_WORKERS = 4           # Increased workers for GPU
RNG_SEED = 42
DROPOUT_RATE = 0.8        # Higher dropout to prevent overfitting
WEIGHT_DECAY = 1e-2       # Much stronger weight decay
LABEL_SMOOTHING = 0.1     # Add label smoothing for better generalization

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='ResNet-based synapse classifier')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=float, default=LR)
args = parser.parse_args()
EPOCHS = args.epochs
LR = args.lr

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
    
    torch.cuda.empty_cache()
else:
    print('No GPU available, using CPU settings')
    BATCH_SIZE = 8
    NUM_WORKERS = 2

# ------------------------- dataset ------------------------------
class Synapse2DDataset(Dataset):
    """
    Dataset class for 2D synapse classification.
    Loads synapse data, takes a 2D slice, and preprocesses it.
    """
    def __init__(self, files: List[str], type_map, data_dir, augment: bool):
        self.files = files
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, data, pre_slice, post_slice):
        if not self.augment:
            return data, pre_slice, post_slice
        # Random horizontal flip
        if random.random() > 0.5:
            data = np.fliplr(data)
            pre_slice = np.fliplr(pre_slice)
            post_slice = np.fliplr(post_slice)
        # Random vertical flip
        if random.random() > 0.5:
            data = np.flipud(data)
            pre_slice = np.flipud(pre_slice)
            post_slice = np.flipud(post_slice)
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            data = np.rot90(data, k)
            pre_slice = np.rot90(pre_slice, k)
            post_slice = np.rot90(post_slice, k)
        return data.copy(), pre_slice.copy(), post_slice.copy()

    def __getitem__(self, idx):
        fname = self.files[idx]
        syn_id = int(fname.split('_')[0])
        syn_type = self.type_map.get(syn_id, 'E')
        label = 1 if syn_type == 'I' else 0

        # Load arrays
        raw_path = os.path.join(self.data_dir, fname)
        pre_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'post_syn_n_mask.npy'))

        raw = np.load(raw_path)
        pre_mask = np.load(pre_path)
        post_mask = np.load(post_path)
        
        combined_mask = np.logical_or(pre_mask, post_mask)
        masked_data = raw * combined_mask

        # Take middle Z slice
        z_mid = masked_data.shape[2] // 2
        data_slice = masked_data[:, :, z_mid]
        pre_slice = pre_mask[:, :, z_mid]
        post_slice = post_mask[:, :, z_mid]
        
        # Bounding box and crop (this part is less important with large resize, but can help focus)
        synapse_pixels = np.where(combined_mask[:, :, z_mid])
        if len(synapse_pixels[0]) > 0:
            min_h, max_h = synapse_pixels[0].min(), synapse_pixels[0].max()
            min_w, max_w = synapse_pixels[1].min(), synapse_pixels[1].max()
            
            padding = 8
            min_h = max(0, min_h - padding)
            max_h = min(data_slice.shape[0], max_h + padding + 1)
            min_w = max(0, min_w - padding)
            max_w = min(data_slice.shape[1], max_w + padding + 1)
            
            data_slice = data_slice[min_h:max_h, min_w:max_w]
            pre_slice = pre_slice[min_h:max_h, min_w:max_w]
            post_slice = post_slice[min_h:max_h, min_w:max_w]
        
        # Resize to fixed size (using cv2 for better interpolation)
        data_slice = cv2.resize(data_slice, (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_AREA)
        pre_slice = cv2.resize(pre_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)
        post_slice = cv2.resize(post_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)

        # Augmentation
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

        return image, label

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

# ------------------------- ResNet model --------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)


def plot_learning_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, e_accs, i_accs, run_timestamp=None):
    """Plot comprehensive learning curves and save to figures directory."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots - 3x2 layout for comprehensive view
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Training and validation loss
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and validation accuracy
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: E and I class accuracies
    ax3 = plt.subplot(3, 2, 3)
    if len(e_accs) > 0:
        ax3.plot(epochs, e_accs, 'g-', label='E (Excitatory) Accuracy', linewidth=2, alpha=0.8)
        ax3.plot(epochs, i_accs, 'm-', label='I (Inhibitory) Accuracy', linewidth=2, alpha=0.8)
        ax3.set_title('Per-Class Validation Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate schedule
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(epochs, learning_rates, 'orange', linewidth=2, alpha=0.8)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Overfitting indicator
    ax5 = plt.subplot(3, 2, 5)
    overfitting_gap = [t - v for t, v in zip(train_accs, val_accs)]
    ax5.plot(epochs, overfitting_gap, 'purple', linewidth=2, alpha=0.8)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    ax5.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    ax5.set_title('Overfitting Indicator (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy Gap (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Training progress summary
    ax6 = plt.subplot(3, 2, 6)
    # Create a summary table
    current_epoch = len(epochs)
    best_val_acc = max(val_accs) if val_accs else 0
    best_epoch = val_accs.index(best_val_acc) + 1 if val_accs else 0
    current_lr = learning_rates[-1] if learning_rates else 0
    
    summary_text = f"""
    Training Progress Summary:
    
    Current Epoch: {current_epoch}
    Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})
    Current Learning Rate: {current_lr:.2e}
    Training Accuracy: {train_accs[-1]:.2f}% (Current)
    Validation Accuracy: {val_accs[-1]:.2f}% (Current)
    Overfitting Gap: {overfitting_gap[-1]:.2f}% (Current)
    
    E Accuracy: {e_accs[-1]:.2f}% (Current)
    I Accuracy: {i_accs[-1]:.2f}% (Current)
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax6.set_title('Training Summary', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Ensure figures directory exists and save with proper error handling
    if run_timestamp is None:
        import datetime
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        os.makedirs('figures', exist_ok=True)
        # Create a proper filename for the plot
        plot_filename = f'figures/resnet_training_curves_{run_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Plot saved successfully to: {plot_filename}')
    except Exception as e:
        print(f'Error saving plot: {e}')
        # Try saving to current directory as fallback
        try:
            plt.savefig('resnet_training_curves.png', dpi=300, bbox_inches='tight')
            print('Plot saved to current directory as fallback')
        except Exception as e2:
            print(f'Failed to save plot even to current directory: {e2}')
    
    plt.close()
    
    # Print progress update
    print(f'Epoch {current_epoch}: Val Acc {val_accs[-1]:.2f}%, Best {best_val_acc:.2f}% (Epoch {best_epoch})')


def main():
    # Setup logging
    logger = setup_logging('resnet')

    # Run start timestamp
    RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ------------------------- data preparation ---------------------
    logger.info('Loading CSV...')
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    data = pd.read_csv(CSV_PATH)
    map_type = {r['id_']: r['pre_clf_type'] for _, r in data.iterrows()}

    all_files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('syn.npy'):
            try:
                syn_id = int(f.split('_')[0])
                if syn_id in map_type and map_type[syn_id] in ['E', 'I']:
                    all_files.append(f)
            except (ValueError, IndexError):
                continue
    E_files = [f for f in all_files if map_type[int(f.split('_')[0])] == 'E']
    I_files = [f for f in all_files if map_type[int(f.split('_')[0])] == 'I']
    size = min(len(E_files), len(I_files))
    if size == 0:
        logger.error("Not enough data for at least one class. Exiting.")
        exit()
    random.shuffle(E_files); random.shuffle(I_files)
    files_bal = E_files[:size] + I_files[:size]
    random.shuffle(files_bal)
    train_f, test_f = train_test_split(files_bal, test_size=0.2, random_state=RNG_SEED,
                                      stratify=[map_type[int(f.split('_')[0])] for f in files_bal])

    # Debug: Print E/I class counts in train/val splits
    train_labels = [map_type[int(f.split('_')[0])] for f in train_f]
    val_labels = [map_type[int(f.split('_')[0])] for f in test_f]
    print('Train E:', train_labels.count('E'), 'I:', train_labels.count('I'))
    print('Val   E:', val_labels.count('E'), 'I:', val_labels.count('I'))

    train_ds = Synapse2DDataset(train_f, map_type, DATA_DIR, augment=True)
    val_ds   = Synapse2DDataset(test_f,  map_type, DATA_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # Debug: Print a batch of labels from DataLoader
    for batch in train_loader:
        _, y = batch
        print('Sample batch labels:', y.tolist())
        break

    labels_train = [1 if map_type[int(f.split('_')[0])] == 'I' else 0 for f in train_f]
    cls_w = compute_class_weight('balanced', classes=np.array([0,1]), y=labels_train)
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=device)
    print('Class weights:', cls_w)

    model = ResNetClassifier().to(device)
    save_path = MODEL_SAVE_PATHS['resnet']
    if args.resume and os.path.exists(save_path):
        logger.info(f'Resuming from checkpoint {save_path}')
        model.load_state_dict(torch.load(save_path, map_location=device))
    logger.info(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    # Print model summary
    try:
        from torchinfo import summary
        print(summary(model, input_size=(BATCH_SIZE, 3, INPUT_XY, INPUT_XY)))
    except ImportError:
        print('Install torchinfo for a model summary (pip install torchinfo)')

    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Warmup scheduler for first 5 epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    # Main scheduler
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    # ------------------------- training loop ------------------------
    best_acc = 0
    best_loss = float('inf')
    patience = 15  # Increased patience for better convergence
    patience_counter = 0
    min_epochs = 20  # Minimum training epochs
    
    # Track learning curves
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    e_accs = []
    i_accs = []
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        tot_loss = tot_corr = tot = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot += y.size(0)
            tot_corr += (out.argmax(1) == y).sum().item()
            pbar.set_postfix(Loss=f'{loss.item():.3f}', Acc=f'{100*tot_corr/tot:.1f}%')
        train_loss = tot_loss / tot
        train_acc = 100*tot_corr/tot

        # ---- validation ----
        model.eval()
        v_tot_loss = v_tot = v_corr = 0
        all_preds, all_lbls = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]')
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_tot_loss += loss.item() * x.size(0)
                v_tot += y.size(0)
                preds = out.argmax(1)
                v_corr += (preds == y).sum().item()
                all_preds.extend(preds.cpu().numpy()); all_lbls.extend(y.cpu().numpy())
                pbar.set_postfix(Loss=f'{loss.item():.3f}', Acc=f'{100*v_corr/v_tot:.1f}%')
        val_loss = v_tot_loss / v_tot
        val_acc = 100*v_corr/v_tot
        
        # Update learning rate
        if epoch <= 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Calculate E and I accuracies
        cm = confusion_matrix(all_lbls, all_preds)
        if cm.shape == (2, 2):
            e_acc = cm[0,0]/cm[0].sum() * 100 if cm[0].sum() > 0 else 0
            i_acc = cm[1,1]/cm[1].sum() * 100 if cm[1].sum() > 0 else 0
        else:
            e_acc = i_acc = 0
        
        # Track E/I accuracies
        if epoch == 1:
            e_accs = [e_acc]
            i_accs = [i_acc]
        else:
            e_accs.append(e_acc)
            i_accs.append(i_acc)
        
        # Update visualization every epoch
        plot_learning_curves(train_losses, train_accs, val_losses, val_accs, 
                           learning_rates, e_accs, i_accs, RUN_TIMESTAMP)
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | GPU Memory: {gpu_memory_used:.1f}GB used, {gpu_memory_cached:.1f}GB cached")
        else:
            logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        cm = confusion_matrix(all_lbls, all_preds)
        logger.info('Confusion Matrix:')
        logger.info(str(cm))
        if cm.shape == (2, 2):
            e_acc = cm[0,0]/cm[0].sum() if cm[0].sum() > 0 else 0
            i_acc = cm[1,1]/cm[1].sum() if cm[1].sum() > 0 else 0
            logger.info(f'E accuracy {e_acc*100:.1f}%  |  I accuracy {i_acc*100:.1f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss # Update best_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f'New best saved ({best_acc:.2f}%)')
        elif val_loss < best_loss: # Consider loss as well
            best_loss = val_loss
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f'New best saved ({best_acc:.2f}%) based on loss')
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
                break

    logger.info(f'Training complete. Best val acc: {best_acc:.2f}%')

    logger.info('Classification report on validation:')
    logger.info(classification_report(all_lbls, all_preds, target_names=['E','I'], zero_division_report=0))

if __name__ == '__main__':
    main() 