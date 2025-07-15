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
INPUT_XY = 224            # Standard input size
EPOCHS = 200              # More epochs for better convergence
LR = 3e-4                 # Higher LR for EfficientNet
NUM_WORKERS = 4           # Increased workers for GPU
RNG_SEED = 42
DROPOUT_RATE = 0.4        # Higher dropout for regularization
WEIGHT_DECAY = 2e-4       # Increased weight decay
LABEL_SMOOTHING = 0.1     # Add label smoothing
MIXUP_ALPHA = 0.2         # Mixup augmentation

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='Advanced synapse classifier targeting 90% accuracy')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet'], help='model architecture')
args = parser.parse_args()
EPOCHS = args.epochs
LR = args.lr

# ------------------------- reproducibility ----------------------
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# GPU optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ------------------------- mixup augmentation ---------------------
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------- focal loss --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------- advanced dataset ------------------------------
class AdvancedSynapseDataset(Dataset):
    """
    Advanced dataset with multi-scale features and heavy augmentation
    """
    def __init__(self, files: List[str], type_map, data_dir, augment: bool):
        self.files = files
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _advanced_augment(self, data, pre_slice, post_slice):
        if not self.augment:
            return data, pre_slice, post_slice
        
        # Geometric augmentations
        if random.random() > 0.5:
            data = np.fliplr(data)
            pre_slice = np.fliplr(pre_slice)
            post_slice = np.fliplr(post_slice)
        
        if random.random() > 0.5:
            data = np.flipud(data)
            pre_slice = np.flipud(pre_slice)
            post_slice = np.flipud(post_slice)
        
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            data = np.rot90(data, k)
            pre_slice = np.rot90(pre_slice, k)
            post_slice = np.rot90(post_slice, k)
        
        # Photometric augmentations
        if random.random() > 0.6:  # Gaussian noise
            noise = np.random.normal(0, 0.02, data.shape)
            data = np.clip(data + noise, 0, 1)
        
        if random.random() > 0.6:  # Brightness
            factor = random.uniform(0.7, 1.3)
            data = np.clip(data * factor, 0, 1)
        
        if random.random() > 0.6:  # Contrast
            mean = data.mean()
            factor = random.uniform(0.7, 1.3)
            data = np.clip((data - mean) * factor + mean, 0, 1)
        
        if random.random() > 0.7:  # Gamma correction
            gamma = random.uniform(0.7, 1.4)
            data = np.power(data, gamma)
        
        if random.random() > 0.8:  # Gaussian blur
            kernel_size = random.choice([3, 5])
            data = cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)
        
        return data.copy(), pre_slice.copy(), post_slice.copy()

    def _extract_multi_scale_features(self, data_slice):
        """Extract features at multiple scales"""
        features = []
        
        # Original data
        features.append(data_slice)
        
        # Edge detection
        edges = cv2.Canny((data_slice * 255).astype(np.uint8), 50, 150) / 255.0
        features.append(edges)
        
        # Gaussian blur for texture
        blurred = cv2.GaussianBlur(data_slice, (5, 5), 0)
        features.append(blurred)
        
        # Sobel gradients
        grad_x = cv2.Sobel(data_slice, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(data_slice, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = np.clip(gradient_mag, 0, 1)
        features.append(gradient_mag)
        
        # Laplacian
        laplacian = cv2.Laplacian(data_slice, cv2.CV_64F)
        laplacian = np.clip(np.abs(laplacian), 0, 1)
        features.append(laplacian)
        
        return features

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

        # Multi-slice approach: use 5 strategic slices
        z_depth = masked_data.shape[2]
        if z_depth >= 5:
            z_indices = [int(z_depth * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]
        elif z_depth >= 3:
            z_indices = [int(z_depth * p) for p in [0.25, 0.5, 0.75]]
            z_indices.extend([z_indices[-1], z_indices[-1]])  # Repeat last
        else:
            z_mid = z_depth // 2
            z_indices = [z_mid] * 5
        
        # Process multiple slices
        data_slices = []
        pre_slices = []
        post_slices = []
        
        for z_idx in z_indices:
            data_slice = masked_data[:, :, z_idx]
            pre_slice = pre_mask[:, :, z_idx]
            post_slice = post_mask[:, :, z_idx]
            
            # Enhanced bounding box
            synapse_pixels = np.where(combined_mask[:, :, z_idx])
            if len(synapse_pixels[0]) > 0:
                min_h, max_h = synapse_pixels[0].min(), synapse_pixels[0].max()
                min_w, max_w = synapse_pixels[1].min(), synapse_pixels[1].max()
                
                # Adaptive padding based on synapse size
                h_size = max_h - min_h
                w_size = max_w - min_w
                padding = max(15, int(0.2 * max(h_size, w_size)))
                
                min_h = max(0, min_h - padding)
                max_h = min(data_slice.shape[0], max_h + padding + 1)
                min_w = max(0, min_w - padding)
                max_w = min(data_slice.shape[1], max_w + padding + 1)
                
                data_slice = data_slice[min_h:max_h, min_w:max_w]
                pre_slice = pre_slice[min_h:max_h, min_w:max_w]
                post_slice = post_slice[min_h:max_h, min_w:max_w]
            
            # Resize
            data_slice = cv2.resize(data_slice, (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_AREA)
            pre_slice = cv2.resize(pre_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)
            post_slice = cv2.resize(post_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)
            
            data_slices.append(data_slice)
            pre_slices.append(pre_slice)
            post_slices.append(post_slice)
        
        # Multi-scale aggregation
        data_slice = np.mean(data_slices, axis=0)
        pre_slice = np.mean(pre_slices, axis=0)
        post_slice = np.mean(post_slices, axis=0)
        
        # Advanced augmentation
        data_slice, pre_slice, post_slice = self._advanced_augment(data_slice, pre_slice, post_slice)

        # Improved normalization
        if data_slice.max() > data_slice.min():
            non_zero_mask = data_slice > 0
            if np.any(non_zero_mask):
                # Robust normalization
                p2, p98 = np.percentile(data_slice[non_zero_mask], [2, 98])
                data_slice = np.clip((data_slice - p2) / (p98 - p2 + 1e-8), 0, 1)
                
                # Histogram equalization (50% chance)
                if random.random() > 0.5 and self.augment:
                    data_slice = cv2.equalizeHist((data_slice * 255).astype(np.uint8)) / 255.0
                
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                if random.random() > 0.7 and self.augment:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    data_slice = clahe.apply((data_slice * 255).astype(np.uint8)) / 255.0
            else:
                data_slice = np.zeros((INPUT_XY, INPUT_XY))
        
        # Extract multi-scale features
        features = self._extract_multi_scale_features(data_slice)
        
        # Combine all channels: [original, edge, blur, gradient, laplacian, pre_mask, post_mask]
        combined_mask_2d = np.logical_or(pre_slice, post_slice).astype(float)
        all_channels = features + [pre_slice, post_slice, combined_mask_2d]
        
        # Stack into 8-channel input
        image = np.stack(all_channels, axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        return image, label

# ------------------------- advanced models --------------------------
class EfficientNetAdvanced(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b4
            self.backbone = efficientnet_b4(pretrained=pretrained)
        except ImportError:
            from torchvision.models import efficientnet_b3
            self.backbone = efficientnet_b3(pretrained=pretrained)
        
        # Modify first layer to accept 8 channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(8, original_conv.out_channels, 
                                                 kernel_size=original_conv.kernel_size,
                                                 stride=original_conv.stride,
                                                 padding=original_conv.padding,
                                                 bias=original_conv.bias)
        
        # Copy pretrained weights for first 3 channels
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = original_conv.weight
            # Initialize new channels with small random values
            nn.init.normal_(self.backbone.features[0][0].weight[:, 3:], 0, 0.005)
        
        # Advanced classifier head
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class ResNetAdvanced(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first layer to accept 8 channels
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(8, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=original_conv.bias)
        
        # Copy pretrained weights for first 3 channels
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = original_conv.weight
            nn.init.normal_(self.resnet.conv1.weight[:, 3:], 0, 0.005)
        
        # Advanced classifier head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# ------------------------- training functions --------------------------
def train_epoch(model, train_loader, criterion, optimizer, epoch, use_mixup=True):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Apply mixup
        if use_mixup and random.random() > 0.3:
            mixed_data, target_a, target_b, lam = mixup_data(data, target, MIXUP_ALPHA)
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        
        if not use_mixup or random.random() <= 0.3:
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * total_correct / total_samples:.2f}%' if total_correct > 0 else 'N/A'
        })
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples if total_correct > 0 else 0
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * total_correct / total_samples:.2f}%'
            })
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy, all_preds, all_targets

def main():
    # Setup logging
    logger = setup_logging('advanced')
    logger.info("Starting Advanced Synapse Classification Training")
    logger.info("=" * 60)
    
    # Run timestamp
    RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ------------------------- data preparation ---------------------
    logger.info('Loading CSV...')
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    
    data = pd.read_csv(CSV_PATH)
    map_type = {r['id_']: r['pre_clf_type'] for _, r in data.iterrows()}

    # Get all valid files
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
    
    logger.info(f'Found {len(E_files)} E files and {len(I_files)} I files')
    
    # Balance dataset
    size = min(len(E_files), len(I_files))
    if size == 0:
        logger.error("Not enough data for at least one class. Exiting.")
        return
    
    random.shuffle(E_files)
    random.shuffle(I_files)
    files_bal = E_files[:size] + I_files[:size]
    random.shuffle(files_bal)
    
    logger.info(f'Using {size} samples per class ({2*size} total)')
    
    # Split data with stratification
    train_f, test_f = train_test_split(
        files_bal, 
        test_size=0.2, 
        random_state=RNG_SEED,
        stratify=[map_type[int(f.split('_')[0])] for f in files_bal]
    )
    
    # Print class distribution
    train_labels = [map_type[int(f.split('_')[0])] for f in train_f]
    val_labels = [map_type[int(f.split('_')[0])] for f in test_f]
    logger.info(f'Train: E={train_labels.count("E")}, I={train_labels.count("I")}')
    logger.info(f'Val: E={val_labels.count("E")}, I={val_labels.count("I")}')

    # Create datasets
    train_ds = AdvancedSynapseDataset(train_f, map_type, DATA_DIR, augment=True)
    val_ds = AdvancedSynapseDataset(test_f, map_type, DATA_DIR, augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True
    )

    # Calculate class weights
    labels_train = [1 if map_type[int(f.split('_')[0])] == 'I' else 0 for f in train_f]
    cls_w = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels_train)
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=device)
    logger.info(f'Class weights: E={cls_w[0]:.3f}, I={cls_w[1]:.3f}')

    # Create model
    if args.model == 'efficientnet':
        model = EfficientNetAdvanced().to(device)
    else:
        model = ResNetAdvanced().to(device)
    
    logger.info(f'Model: {args.model}')
    logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss function and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=cls_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR, 
        epochs=EPOCHS, 
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000
    )

    # ------------------------- training loop ------------------------
    best_acc = 0
    best_epoch = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validation
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate per-class accuracy
        cm = confusion_matrix(val_targets, val_preds)
        if cm.shape == (2, 2):
            e_acc = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
            i_acc = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
        else:
            e_acc = i_acc = 0
        
        logger.info(f'Epoch {epoch}/{EPOCHS}:')
        logger.info(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        logger.info(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        logger.info(f'  E Acc: {e_acc:.2f}%, I Acc: {i_acc:.2f}%')
        logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_advanced_model_{RUN_TIMESTAMP}.pth')
            logger.info(f'  🎯 NEW BEST: {best_acc:.2f}% (saved)')
        
        # Early success check
        if val_acc >= 90.0:
            logger.info(f'🎉 SUCCESS! Reached 90% accuracy at epoch {epoch}')
            break
        
        logger.info('-' * 60)

    logger.info(f'Training complete!')
    logger.info(f'Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch})')
    
    # Final evaluation
    logger.info('\nFinal Classification Report:')
    logger.info(classification_report(val_targets, val_preds, target_names=['E', 'I']))
    
    logger.info('\nFinal Confusion Matrix:')
    logger.info(str(confusion_matrix(val_targets, val_preds)))

if __name__ == '__main__':
    main() 