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
BATCH_SIZE = 64           # Larger batch size for faster epochs
INPUT_XY = 224            # Standard ResNet input size
EPOCHS = 100              # More epochs for better convergence
LR = 2e-6                 # Lower LR to prevent overfitting
NUM_WORKERS = 4           # Increased workers for GPU
RNG_SEED = 42
DROPOUT_RATE = 0.3        # Lower dropout to reduce underfitting
WEIGHT_DECAY = 1e-4       # Lower weight decay to reduce underfitting
LABEL_SMOOTHING = 0       # Remove label smoothing

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
    Loads synapse data, uses all Z-slices as individual samples.
    """
    def __init__(self, files: List[str], type_map, data_dir, augment: bool):
        self.files = files
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment
        self.all_samples = []  # Store all possible samples
        
        # Generate all possible Z-slice samples
        for fname in files:
            syn_id = int(fname.split('_')[0])
            syn_type = self.type_map.get(syn_id, 'E')
            label = 1 if syn_type == 'I' else 0
            
            # Load to get Z dimension
            raw_path = os.path.join(self.data_dir, fname)
            raw = np.load(raw_path)
            z_depth = raw.shape[2]
            
            # Add all Z-slices as possible samples
            for z_idx in range(z_depth):
                self.all_samples.append((fname, z_idx, syn_id, label))
        
        # Separate E and I samples
        self.e_samples = [s for s in self.all_samples if s[3] == 0]  # E class
        self.i_samples = [s for s in self.all_samples if s[3] == 1]  # I class
        
        # Subsample to fixed size (e.g., 3000 samples per epoch)
        self.epoch_size = min(3000, len(self.all_samples))
        self.resample_epoch()
    
    def resample_epoch(self):
        """Randomly sample a balanced subset for this epoch."""
        # Calculate how many samples per class to maintain balance
        samples_per_class = self.epoch_size // 2
        
        # Sample equal numbers from each class
        e_subset = random.sample(self.e_samples, min(samples_per_class, len(self.e_samples)))
        i_subset = random.sample(self.i_samples, min(samples_per_class, len(self.i_samples)))
        
        # Combine and shuffle
        self.samples = e_subset + i_subset
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _augment(self, data, pre_slice, post_slice):
        # No augmentation - return original data
        return data, pre_slice, post_slice

    def __getitem__(self, idx):
        fname, z_idx, syn_id, label = self.samples[idx]

        # Load arrays
        raw_path = os.path.join(self.data_dir, fname)
        pre_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'post_syn_n_mask.npy'))

        raw = np.load(raw_path)
        pre_mask = np.load(pre_path)
        post_mask = np.load(post_path)
        
        combined_mask = np.logical_or(pre_mask, post_mask)
        masked_data = raw * combined_mask

        # Take specific Z slice
        data_slice = masked_data[:, :, z_idx]
        pre_slice = pre_mask[:, :, z_idx]
        post_slice = post_mask[:, :, z_idx]
        
        # Skip slices with no synapse data
        if not np.any(combined_mask[:, :, z_idx]):
            # Return zeros for empty slices
            image = np.zeros((3, INPUT_XY, INPUT_XY), dtype=np.float32)
            return torch.from_numpy(image), label
        
        # Bounding box and crop
        synapse_pixels = np.where(combined_mask[:, :, z_idx])
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
        
        # Resize to fixed size
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
            else:
                data_slice = np.zeros((INPUT_XY, INPUT_XY))
        
        image = np.stack([data_slice, pre_slice, post_slice], axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        return image, label

# Add function to evaluate per-synapse accuracy
def evaluate_per_synapse(model, dataset, device):
    """Evaluate per-synapse accuracy using confidence-weighted voting across Z-slices."""
    model.eval()
    synapse_predictions = {}  # syn_id -> list of (prediction, confidence)
    synapse_labels = {}       # syn_id -> true label
    
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            fname, z_idx, syn_id, _ = dataset.samples[i]
            
            image = image.unsqueeze(0).to(device)
            output = model(image)
            
            # Get softmax probabilities for confidence weighting
            probs = F.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = probs[0, pred].item()  # Confidence is the max probability
            
            if syn_id not in synapse_predictions:
                synapse_predictions[syn_id] = []
                synapse_labels[syn_id] = label
            
            synapse_predictions[syn_id].append((pred, confidence))
    
    # Confidence-weighted vote for each synapse
    correct = 0
    total = 0
    for syn_id in synapse_predictions:
        pred_conf_pairs = synapse_predictions[syn_id]
        true_label = synapse_labels[syn_id]
        
        # Calculate confidence-weighted scores for each class
        class_0_score = 0.0  # E class
        class_1_score = 0.0  # I class
        
        for pred, conf in pred_conf_pairs:
            if pred == 0:
                class_0_score += conf
            else:
                class_1_score += conf
        
        # Final prediction is the class with higher confidence-weighted score
        final_pred = 1 if class_1_score > class_0_score else 0
        
        if final_pred == true_label:
            correct += 1
        total += 1
    
    return 100 * correct / total if total > 0 else 0

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
        self.resnet = models.resnet50(pretrained=False)  # Changed to ResNet50 for higher capacity
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


def plot_learning_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, e_accs, i_accs, 
                        synapse_train_accs, synapse_val_accs, run_timestamp=None):
    """Plot comprehensive learning curves including per-synapse accuracies."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots - 3x3 layout for comprehensive view
    fig = plt.figure(figsize=(24, 18))
    
    # Plot 1: Training and validation loss
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-slice accuracy
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy (Per-Slice)', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy (Per-Slice)', linewidth=2, alpha=0.8)
    ax2.set_title('Per-Slice Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Per-synapse accuracy
    ax3 = plt.subplot(3, 3, 3)
    if len(synapse_train_accs) > 0:
        ax3.plot(epochs, synapse_train_accs, 'g-', label='Training Accuracy (Per-Synapse)', linewidth=2, alpha=0.8)
        ax3.plot(epochs, synapse_val_accs, 'm-', label='Validation Accuracy (Per-Synapse)', linewidth=2, alpha=0.8)
        ax3.set_title('Per-Synapse Accuracy (Majority Vote)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: E and I class accuracies
    ax4 = plt.subplot(3, 3, 4)
    if len(e_accs) > 0:
        ax4.plot(epochs, e_accs, 'g-', label='E (Excitatory) Accuracy', linewidth=2, alpha=0.8)
        ax4.plot(epochs, i_accs, 'm-', label='I (Inhibitory) Accuracy', linewidth=2, alpha=0.8)
        ax4.set_title('Per-Class Validation Accuracy', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Learning rate schedule
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(epochs, learning_rates, 'orange', linewidth=2, alpha=0.8)
    ax5.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Overfitting indicator (per-slice)
    ax6 = plt.subplot(3, 3, 6)
    overfitting_gap = [t - v for t, v in zip(train_accs, val_accs)]
    ax6.plot(epochs, overfitting_gap, 'purple', linewidth=2, alpha=0.8)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    ax6.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    ax6.set_title('Overfitting Indicator (Per-Slice)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy Gap (%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Synapse overfitting indicator
    ax7 = plt.subplot(3, 3, 7)
    if len(synapse_train_accs) > 0:
        synapse_overfitting_gap = [t - v for t, v in zip(synapse_train_accs, synapse_val_accs)]
        ax7.plot(epochs, synapse_overfitting_gap, 'brown', linewidth=2, alpha=0.8)
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
        ax7.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
        ax7.set_title('Overfitting Indicator (Per-Synapse)', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Accuracy Gap (%)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Comparison of slice vs synapse accuracy
    ax8 = plt.subplot(3, 3, 8)
    if len(synapse_val_accs) > 0:
        ax8.plot(epochs, val_accs, 'r-', label='Per-Slice Val Accuracy', linewidth=2, alpha=0.8)
        ax8.plot(epochs, synapse_val_accs, 'm-', label='Per-Synapse Val Accuracy', linewidth=2, alpha=0.8)
        ax8.set_title('Slice vs Synapse Validation Accuracy', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Accuracy (%)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Training progress summary
    ax9 = plt.subplot(3, 3, 9)
    current_epoch = len(epochs)
    best_val_acc = max(val_accs) if val_accs else 0
    best_epoch = val_accs.index(best_val_acc) + 1 if val_accs else 0
    best_synapse_acc = max(synapse_val_accs) if synapse_val_accs else 0
    current_lr = learning_rates[-1] if learning_rates else 0
    
    summary_text = f"""
    Training Progress Summary:
    
    Current Epoch: {current_epoch}
    Best Per-Slice Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})
    Best Per-Synapse Val Accuracy: {best_synapse_acc:.2f}%
    Current Learning Rate: {current_lr:.2e}
    
    Current Per-Slice Accuracy: {val_accs[-1]:.2f}%
    Current Per-Synapse Accuracy: {synapse_val_accs[-1]:.2f}%
    
    E Accuracy: {e_accs[-1]:.2f}% (Current)
    I Accuracy: {i_accs[-1]:.2f}% (Current)
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.set_title('Training Summary', fontsize=14, fontweight='bold')
    ax9.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if run_timestamp is None:
        import datetime
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        os.makedirs('figures', exist_ok=True)
        plot_filename = f'figures/resnet_training_curves_{run_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Plot saved successfully to: {plot_filename}')
    except Exception as e:
        print(f'Error saving plot: {e}')
        try:
            plt.savefig('resnet_training_curves.png', dpi=300, bbox_inches='tight')
            print('Plot saved to current directory as fallback')
        except Exception as e2:
            print(f'Failed to save plot even to current directory: {e2}')
    
    plt.close()
    
    # Print progress update
    print(f'Epoch {current_epoch}: Per-Slice Val Acc {val_accs[-1]:.2f}%, Per-Synapse Val Acc {synapse_val_accs[-1]:.2f}%')


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
    
    # Replace scheduler with ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

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
    synapse_train_accs = []
    synapse_val_accs = []
    
    for epoch in range(1, EPOCHS+1):
        # Resample training data for this epoch
        train_ds.resample_epoch()
        
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
        
        # Update learning rate with ReduceLROnPlateau
        scheduler.step(val_loss)
        
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
        
        # Evaluate per-synapse accuracy
        synapse_train_acc = evaluate_per_synapse(model, train_ds, device)
        synapse_val_acc = evaluate_per_synapse(model, val_ds, device)

        # Track per-synapse accuracies
        if epoch == 1:
            synapse_train_accs = [synapse_train_acc]
            synapse_val_accs = [synapse_val_acc]
        else:
            synapse_train_accs.append(synapse_train_acc)
            synapse_val_accs.append(synapse_val_acc)
        
        # Update visualization every epoch
        plot_learning_curves(train_losses, train_accs, val_losses, val_accs, 
                           learning_rates, e_accs, i_accs, synapse_train_accs, synapse_val_accs, RUN_TIMESTAMP)
        
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

        # Evaluate per-synapse accuracy
        # synapse_train_acc = evaluate_per_synapse(model, train_ds, device)
        # synapse_val_acc = evaluate_per_synapse(model, val_ds, device)

        # Track per-synapse accuracies
        # if epoch == 1:
        #     synapse_train_accs = [synapse_train_acc]
        #     synapse_val_accs = [synapse_val_acc]
        # else:
        #     synapse_train_accs.append(synapse_train_acc)
        #     synapse_val_accs.append(synapse_val_acc)

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
            # Remove early stopping: always run for EPOCHS
            # if patience_counter >= patience and epoch >= min_epochs:
            #     logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            #     break

    logger.info(f'Training complete. Best val acc: {best_acc:.2f}%')

    logger.info('Classification report on validation:')
    logger.info(classification_report(all_lbls, all_preds, target_names=['E','I'], zero_division_report=0))

if __name__ == '__main__':
    main() 