import os
import random
import warnings
import argparse
from typing import List, Tuple, Dict
import math

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

# ------------------------- enhanced configuration -------------------------
BATCH_SIZE = 16           # Reduced for more stable training
INPUT_XY = 256            # Larger input for better features
EPOCHS = 300              # More epochs for convergence  
LR = 1e-4                 # Lower starting LR
NUM_WORKERS = 4           
RNG_SEED = 42
DROPOUT_RATE = 0.5        # Higher dropout for regularization
WEIGHT_DECAY = 5e-4       # Increased weight decay
LABEL_SMOOTHING = 0.15    # More label smoothing
MIXUP_ALPHA = 0.4         # More aggressive mixup
CUTMIX_ALPHA = 1.0        # CutMix augmentation
PROGRESSIVE_RESIZE_EPOCHS = [0, 50, 100, 150]  # Progressive resizing schedule
PROGRESSIVE_SIZES = [128, 192, 224, 256]        # Corresponding sizes
TTA_FLIPS = 8             # Test-time augmentation flips

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='Ultra-Advanced synapse classifier targeting 90% accuracy')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet', 'ensemble'], help='model architecture')
parser.add_argument('--tta', action='store_true', help='use test-time augmentation')
parser.add_argument('--progressive', action='store_true', help='use progressive resizing')
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
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

# ------------------------- augmentation utilities ---------------------
def mixup_data(x, y, alpha=1.0):
    """Enhanced Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to match the actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------- enhanced focal loss --------------------------
class EnhancedFocalLoss(nn.Module):
    """Enhanced Focal Loss with class balancing"""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
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

# ------------------------- attention mechanisms --------------------------
class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ------------------------- ultra-advanced dataset ------------------------------
class UltraAdvancedSynapseDataset(Dataset):
    """
    Ultra-advanced dataset with progressive resizing, multi-scale features, and heavy augmentation
    """
    def __init__(self, files: List[str], type_map, data_dir, augment: bool, current_size: int = INPUT_XY):
        self.files = files
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment
        self.current_size = current_size

    def __len__(self):
        return len(self.files)
    
    def update_size(self, new_size: int):
        """Update the current image size for progressive resizing"""
        self.current_size = new_size
        print(f"Updated dataset size to {new_size}x{new_size}")

    def _ultra_augment(self, data, pre_slice, post_slice):
        """Ultra-aggressive augmentation"""
        if not self.augment:
            return data, pre_slice, post_slice
        
        # Geometric augmentations (more aggressive)
        if random.random() > 0.4:
            data = np.fliplr(data)
            pre_slice = np.fliplr(pre_slice)
            post_slice = np.fliplr(post_slice)
        
        if random.random() > 0.4:
            data = np.flipud(data)
            pre_slice = np.flipud(pre_slice)
            post_slice = np.flipud(post_slice)
        
        if random.random() > 0.3:
            k = random.choice([1, 2, 3])
            data = np.rot90(data, k)
            pre_slice = np.rot90(pre_slice, k)
            post_slice = np.rot90(post_slice, k)
        
        # Photometric augmentations (more aggressive)
        if random.random() > 0.5:  # Gaussian noise
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, data.shape)
            data = np.clip(data + noise, 0, 1)
        
        if random.random() > 0.5:  # Brightness
            factor = random.uniform(0.6, 1.4)
            data = np.clip(data * factor, 0, 1)
        
        if random.random() > 0.5:  # Contrast
            mean = data.mean()
            factor = random.uniform(0.6, 1.4)
            data = np.clip((data - mean) * factor + mean, 0, 1)
        
        if random.random() > 0.6:  # Gamma correction
            gamma = random.uniform(0.5, 1.8)
            data = np.power(data, gamma)
        
        if random.random() > 0.7:  # Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            data = cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)
        
        # New augmentations
        if random.random() > 0.8:  # Salt and pepper noise
            noise = np.random.random(data.shape)
            data[noise < 0.05] = 0
            data[noise > 0.95] = 1
        
        if random.random() > 0.8:  # Elastic deformation (safer version)
            try:
                alpha = random.uniform(0, 20)  # Reduced intensity
                sigma = random.uniform(2, 5)   # More conservative sigma
                if alpha > 0 and sigma > 0 and data.shape[0] < 1000 and data.shape[1] < 1000:  # Size check
                    # Create displacement fields
                    dx = cv2.GaussianBlur((np.random.random(data.shape) - 0.5), (0, 0), sigma) * alpha
                    dy = cv2.GaussianBlur((np.random.random(data.shape) - 0.5), (0, 0), sigma) * alpha
                    
                    # Create coordinate grids
                    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
                    
                    # Apply displacement with bounds checking
                    map_x = (x + dx).astype(np.float32)
                    map_y = (y + dy).astype(np.float32)
                    
                    # Ensure coordinates are within valid range
                    map_x = np.clip(map_x, 0, data.shape[1] - 1)
                    map_y = np.clip(map_y, 0, data.shape[0] - 1)
                    
                    data = cv2.remap(data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            except:
                # If elastic deformation fails, skip it
                pass
        
        return data.copy(), pre_slice.copy(), post_slice.copy()

    def _extract_advanced_features(self, data_slice):
        """Extract advanced multi-scale features with more sophisticated methods"""
        features = []
        
        # Original data
        features.append(data_slice)
        
        # Multi-scale edge detection
        edges_canny = cv2.Canny((data_slice * 255).astype(np.uint8), 50, 150) / 255.0
        features.append(edges_canny)
        
        # Multi-scale Gaussian pyramids
        pyramid_scales = [1, 0.7, 0.5]
        for scale in pyramid_scales[1:]:  # Skip original scale
            h, w = data_slice.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(data_slice, (new_w, new_h))
            scaled = cv2.resize(scaled, (w, h))  # Resize back
            features.append(scaled)
        
        # Gradient features
        grad_x = cv2.Sobel(data_slice, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(data_slice, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = np.clip(gradient_mag, 0, 1)
        features.append(gradient_mag)
        
        # Laplacian of Gaussian
        laplacian = cv2.Laplacian(data_slice, cv2.CV_64F)
        laplacian = np.clip(np.abs(laplacian), 0, 1)
        features.append(laplacian)
        
        # Structure tensor
        Ixx = cv2.Sobel(data_slice, cv2.CV_64F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(data_slice, cv2.CV_64F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        
        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy
        harris = det - 0.04 * trace**2
        harris = np.clip(np.abs(harris), 0, 1)
        features.append(harris)
        
        return features[:8]  # Limit to 8 features to match original

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

        # Enhanced multi-slice approach: adaptive slice selection
        z_depth = masked_data.shape[2]
        if z_depth >= 7:
            # Use 7 strategic slices for better representation
            z_indices = [int(z_depth * p) for p in [0.15, 0.28, 0.4, 0.5, 0.6, 0.72, 0.85]]
        elif z_depth >= 5:
            z_indices = [int(z_depth * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]
            z_indices.extend([z_indices[-1], z_indices[-1]])  # Pad to 7
        elif z_depth >= 3:
            z_indices = [int(z_depth * p) for p in [0.25, 0.5, 0.75]]
            z_indices.extend([z_indices[-1]] * 4)  # Pad to 7
        else:
            z_mid = z_depth // 2
            z_indices = [z_mid] * 7
        
        # Process multiple slices with weighted averaging
        data_slices = []
        pre_slices = []
        post_slices = []
        weights = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]  # Center-weighted
        
        for i, z_idx in enumerate(z_indices):
            data_slice = masked_data[:, :, z_idx]
            pre_slice = pre_mask[:, :, z_idx]
            post_slice = post_mask[:, :, z_idx]
            
            # Enhanced bounding box with adaptive padding
            synapse_pixels = np.where(combined_mask[:, :, z_idx])
            if len(synapse_pixels[0]) > 0:
                min_h, max_h = synapse_pixels[0].min(), synapse_pixels[0].max()
                min_w, max_w = synapse_pixels[1].min(), synapse_pixels[1].max()
                
                # Adaptive padding based on synapse size and slice importance
                h_size = max_h - min_h
                w_size = max_w - min_w
                base_padding = max(20, int(0.25 * max(h_size, w_size)))
                weight_factor = weights[i] * 2  # More padding for important slices
                padding = int(base_padding * (1 + weight_factor))
                
                min_h = max(0, min_h - padding)
                max_h = min(data_slice.shape[0], max_h + padding + 1)
                min_w = max(0, min_w - padding)
                max_w = min(data_slice.shape[1], max_w + padding + 1)
                
                data_slice = data_slice[min_h:max_h, min_w:max_w]
                pre_slice = pre_slice[min_h:max_h, min_w:max_w]
                post_slice = post_slice[min_h:max_h, min_w:max_w]
            
            # Resize to current size (progressive resizing)
            data_slice = cv2.resize(data_slice, (self.current_size, self.current_size), interpolation=cv2.INTER_AREA)
            pre_slice = cv2.resize(pre_slice.astype(float), (self.current_size, self.current_size), interpolation=cv2.INTER_NEAREST)
            post_slice = cv2.resize(post_slice.astype(float), (self.current_size, self.current_size), interpolation=cv2.INTER_NEAREST)
            
            data_slices.append(data_slice * weights[i])
            pre_slices.append(pre_slice * weights[i])
            post_slices.append(post_slice * weights[i])
        
        # Weighted aggregation
        data_slice = np.sum(data_slices, axis=0)
        pre_slice = np.sum(pre_slices, axis=0)
        post_slice = np.sum(post_slices, axis=0)
        
        # Ultra-aggressive augmentation
        data_slice, pre_slice, post_slice = self._ultra_augment(data_slice, pre_slice, post_slice)

        # Advanced normalization with outlier removal
        if data_slice.max() > data_slice.min():
            non_zero_mask = data_slice > 0
            if np.any(non_zero_mask):
                # Robust normalization with outlier clipping
                p1, p99 = np.percentile(data_slice[non_zero_mask], [1, 99])
                data_slice = np.clip((data_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
                
                # Advanced histogram equalization
                if random.random() > 0.6 and self.augment:
                    # Adaptive histogram equalization
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    data_slice = clahe.apply((data_slice * 255).astype(np.uint8)) / 255.0
                elif random.random() > 0.4 and self.augment:
                    # Standard histogram equalization
                    data_slice = cv2.equalizeHist((data_slice * 255).astype(np.uint8)) / 255.0
            else:
                data_slice = np.zeros((self.current_size, self.current_size))
        
        # Extract advanced multi-scale features
        features = self._extract_advanced_features(data_slice)
        
        # Combine all channels with enhanced mask features
        combined_mask_2d = np.logical_or(pre_slice, post_slice).astype(float)
        
        # Create final feature stack (ensure exactly 8 channels)
        if len(features) >= 6:
            all_channels = features[:6] + [pre_slice, post_slice]
        else:
            all_channels = features + [pre_slice, post_slice]
            while len(all_channels) < 8:
                all_channels.append(combined_mask_2d)
        
        # Stack into 8-channel input
        image = np.stack(all_channels[:8], axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        return image, label

# ------------------------- ensemble models --------------------------
class UltraEfficientNet(nn.Module):
    """Ultra-enhanced EfficientNet with attention and advanced features"""
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
        
        # Copy pretrained weights for first 3 channels and initialize others
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = original_conv.weight
            # Initialize new channels with Xavier normal
            nn.init.xavier_normal_(self.backbone.features[0][0].weight[:, 3:])
        
        # Get feature size
        original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 8, 256, 256)
            features = self.backbone(dummy_input)
            num_features = features.shape[1]
        
        # Ultra-advanced classifier head with residual connections
        self.backbone.classifier = self._build_advanced_classifier(num_features, num_classes)

    def _build_advanced_classifier(self, num_features, num_classes):
        """Build an advanced classifier with residual connections and attention"""
        return nn.Sequential(
            # EfficientNet already has AdaptiveAvgPool2d and Flatten, so start directly with Linear
            
            # First residual block
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            # Second residual block
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            # Third block
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 2),
            
            # Final classification
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class UltraResNet(nn.Module):
    """Ultra-enhanced ResNet with attention mechanisms"""
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
        
        # Copy pretrained weights and initialize new channels
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = original_conv.weight
            nn.init.xavier_normal_(self.resnet.conv1.weight[:, 3:])
        
        # Add attention modules
        self.attention1 = CBAM(256)  # After layer1
        self.attention2 = CBAM(512)  # After layer2
        self.attention3 = CBAM(1024) # After layer3
        self.attention4 = CBAM(2048) # After layer4
        
        # Ultra-advanced classifier
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = self._build_advanced_classifier(num_ftrs, num_classes)
    
    def _build_advanced_classifier(self, num_features, num_classes):
        """Build advanced classifier with multiple pathways"""
        return nn.Sequential(
            # ResNet forward method already does avgpool and flatten, so start with Linear
            
            # Multi-scale feature processing
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 4),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.attention1(x)
        
        x = self.resnet.layer2(x)
        x = self.attention2(x)
        
        x = self.resnet.layer3(x)
        x = self.attention3(x)
        
        x = self.resnet.layer4(x)
        x = self.attention4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

# Test-time augmentation
def test_time_augmentation(model, data, num_tta=TTA_FLIPS):
    """Apply test-time augmentation for better accuracy"""
    model.eval()
    with torch.no_grad():
        predictions = []
        
        # Original prediction
        pred = model(data)
        predictions.append(F.softmax(pred, dim=1))
        
        # Horizontal flip
        if num_tta >= 2:
            pred = model(torch.flip(data, [3]))
            predictions.append(F.softmax(pred, dim=1))
        
        # Vertical flip
        if num_tta >= 3:
            pred = model(torch.flip(data, [2]))
            predictions.append(F.softmax(pred, dim=1))
        
        # Both flips
        if num_tta >= 4:
            pred = model(torch.flip(data, [2, 3]))
            predictions.append(F.softmax(pred, dim=1))
        
        # Rotations
        if num_tta >= 8:
            for angle in [90, 180, 270]:
                k = angle // 90
                rotated = torch.rot90(data, k, [2, 3])
                pred = model(rotated)
                predictions.append(F.softmax(pred, dim=1))
                
                # Also flip the rotation
                pred = model(torch.flip(rotated, [3]))
                predictions.append(F.softmax(pred, dim=1))
        
        # Average all predictions
        final_pred = torch.stack(predictions).mean(0)
        return final_pred

# ------------------------- enhanced plotting --------------------------
def plot_learning_curves(train_losses, train_accs, val_losses, val_accs, learning_rates, e_accs, i_accs, run_timestamp=None):
    """Plot comprehensive learning curves and save to figures directory."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots - 3x2 layout for comprehensive view
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Training and validation loss
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(epochs, train_losses, 'orange', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'purple', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and validation accuracy
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(epochs, train_accs, 'orange', label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_accs, 'purple', label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: E and I class accuracies
    ax3 = plt.subplot(3, 2, 3)
    if len(e_accs) > 0:
        ax3.plot(epochs, e_accs, 'red', label='E (Excitatory) Accuracy', linewidth=2, alpha=0.8)
        ax3.plot(epochs, i_accs, 'blue', label='I (Inhibitory) Accuracy', linewidth=2, alpha=0.8)
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
    current_epoch = len(epochs)
    best_val_acc = max(val_accs) if val_accs else 0
    best_epoch = val_accs.index(best_val_acc) + 1 if val_accs else 0
    current_lr = learning_rates[-1] if learning_rates else 0
    
    summary_text = f"""
    🎯 ADVANCED SYNAPSE CLASSIFIER
    
    Current Epoch: {current_epoch}
    Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})
    Current Learning Rate: {current_lr:.2e}
    Training Accuracy: {train_accs[-1]:.2f}% (Current)
    Validation Accuracy: {val_accs[-1]:.2f}% (Current)
    Overfitting Gap: {overfitting_gap[-1]:.2f}% (Current)
    
    E Accuracy: {e_accs[-1]:.2f}% (Current)
    I Accuracy: {i_accs[-1]:.2f}% (Current)
    
    🎯 TARGET: 90% Accuracy
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax6.set_title('Training Summary', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        os.makedirs('figures', exist_ok=True)
        plot_filename = f'figures/advanced_training_curves_{run_timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'📊 Plot saved: {plot_filename}')
    except Exception as e:
        print(f'❌ Error saving plot: {e}')
    
    plt.close()

# ------------------------- enhanced training functions --------------------------
def train_epoch(model, train_loader, criterion, optimizer, epoch, use_mixup=True, use_cutmix=True):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Apply augmentations
        if use_mixup and random.random() < 0.5:
            # Mixup augmentation
            mixed_data, y_a, y_b, lam = mixup_data(data, target, MIXUP_ALPHA)
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        elif use_cutmix and random.random() < 0.5:
            # CutMix augmentation
            mixed_data, y_a, y_b, lam = cutmix_data(data, target, CUTMIX_ALPHA)
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            # Standard training
            output = model(data)
            loss = criterion(output, target)
        
        loss.backward()
        
        # Gradient clipping with adaptive threshold
        max_norm = 2.0 if epoch < 50 else 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        
        # Calculate accuracy (use original labels for mixed data)
        pred = output.argmax(dim=1)
        if use_mixup or use_cutmix:
            # For mixed data, use original target for accuracy calculation
            total_correct += pred.eq(target).sum().item()
        else:
            total_correct += pred.eq(target).sum().item()
            
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * total_correct / total_samples:.2f}%'
        })
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, epoch, use_tta=False):
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
            
            if use_tta and epoch % 10 == 0:  # Use TTA every 10 epochs
                # Test-time augmentation
                output = test_time_augmentation(model, data, num_tta=TTA_FLIPS)
                loss = criterion(torch.log(output + 1e-8), target)  # Convert softmax back to logits
            else:
                # Standard prediction
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if use_tta and epoch % 10 == 0:
                pred = output.argmax(dim=1)
            else:
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

# Ensemble training functions
def train_epoch_ensemble(models, train_loader, criterion, optimizers, epoch):
    """Train ensemble of models"""
    for model in models:
        model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train Ensemble]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        ensemble_outputs = []
        losses = []
        
        # Train each model in ensemble
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            
            # Apply different augmentations to each model
            if i == 0 and random.random() < 0.5:
                # Mixup for first model
                mixed_data, y_a, y_b, lam = mixup_data(data, target, MIXUP_ALPHA)
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            elif i == 1 and random.random() < 0.5:
                # CutMix for second model
                mixed_data, y_a, y_b, lam = cutmix_data(data, target, CUTMIX_ALPHA)
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                # Standard training
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            ensemble_outputs.append(F.softmax(output, dim=1))
            losses.append(loss.item())
        
        # Average ensemble predictions for accuracy calculation
        ensemble_output = torch.stack(ensemble_outputs).mean(0)
        pred = ensemble_output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        
        # Average losses
        avg_loss = sum(losses) / len(losses)
        total_loss += avg_loss * data.size(0)
        total_samples += data.size(0)
        
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{100.0 * total_correct / total_samples:.2f}%'
        })
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy

def validate_epoch_ensemble(models, val_loader, criterion, epoch):
    """Validate ensemble of models"""
    for model in models:
        model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val Ensemble]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            ensemble_outputs = []
            losses = []
            
            # Get predictions from each model
            for model in models:
                output = model(data)
                loss = criterion(output, target)
                ensemble_outputs.append(F.softmax(output, dim=1))
                losses.append(loss.item())
            
            # Average ensemble predictions
            ensemble_output = torch.stack(ensemble_outputs).mean(0)
            pred = ensemble_output.argmax(dim=1)
            
            total_correct += pred.eq(target).sum().item()
            total_loss += sum(losses) / len(losses) * data.size(0)
            total_samples += data.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{sum(losses) / len(losses):.4f}',
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
    train_ds = UltraAdvancedSynapseDataset(train_f, map_type, DATA_DIR, augment=True)
    val_ds = UltraAdvancedSynapseDataset(test_f, map_type, DATA_DIR, augment=False)

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
        model = UltraEfficientNet().to(device)
    elif args.model == 'resnet':
        model = UltraResNet().to(device)
    else:  # ensemble
        # For ensemble, create both models and average their predictions
        model1 = UltraEfficientNet().to(device)
        model2 = UltraResNet().to(device)
        model = [model1, model2]
    
    logger.info(f'Model: {args.model}')
    if isinstance(model, list):
        total_params = sum(sum(p.numel() for p in m.parameters()) for m in model)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {total_params:,}')

    # Loss function and optimizer
    criterion = EnhancedFocalLoss(alpha=0.25, gamma=2.0, weight=cls_w)
    
    if isinstance(model, list):
        # For ensemble, create optimizers for both models
        optimizer1 = optim.AdamW(model[0].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer2 = optim.AdamW(model[1].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = [optimizer1, optimizer2]
        
        # Schedulers for both models
        scheduler1 = optim.lr_scheduler.OneCycleLR(
            optimizer1, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader),
            pct_start=0.3, div_factor=25, final_div_factor=10000
        )
        scheduler2 = optim.lr_scheduler.OneCycleLR(
            optimizer2, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader),
            pct_start=0.3, div_factor=25, final_div_factor=10000
        )
        scheduler = [scheduler1, scheduler2]
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Enhanced learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LR, 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000
        )

    # ------------------------- enhanced training loop ------------------------
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 20  # Early stopping patience
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    e_accs = []
    i_accs = []
    
    for epoch in range(1, EPOCHS + 1):
        # Progressive resizing
        if args.progressive and epoch in PROGRESSIVE_RESIZE_EPOCHS:
            new_size = PROGRESSIVE_SIZES[PROGRESSIVE_RESIZE_EPOCHS.index(epoch)]
            train_ds.update_size(new_size)
            val_ds.update_size(new_size)
            logger.info(f'Epoch {epoch}: Progressive resizing to {new_size}x{new_size}')

        # Training with ensemble support
        if isinstance(model, list):
            # Ensemble training
            train_loss, train_acc = train_epoch_ensemble(model, train_loader, criterion, optimizer, epoch)
        else:
            # Single model training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validation with ensemble support
        if isinstance(model, list):
            # Ensemble validation
            val_loss, val_acc, val_preds, val_targets = validate_epoch_ensemble(model, val_loader, criterion, epoch)
        else:
            # Single model validation with optional TTA
            val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, epoch, use_tta=args.tta)
        
        # Update schedulers
        if isinstance(scheduler, list):
            for sched in scheduler:
                sched.step()
        else:
            scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Track learning rates
        if isinstance(optimizer, list):
            learning_rates.append(optimizer[0].param_groups[0]['lr'])
        else:
            learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Calculate per-class accuracy
        cm = confusion_matrix(val_targets, val_preds)
        if cm.shape == (2, 2):
            e_acc = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
            i_acc = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
        else:
            e_acc = i_acc = 0
        
        e_accs.append(e_acc)
        i_accs.append(i_acc)
        
        # Generate learning curves every 5 epochs or when we hit a new best
        if epoch % 5 == 0 or val_acc > best_acc:
            plot_learning_curves(train_losses, train_accs, val_losses, val_accs, 
                               learning_rates, e_accs, i_accs, RUN_TIMESTAMP)
        
        logger.info(f'Epoch {epoch}/{EPOCHS}:')
        logger.info(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        logger.info(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        logger.info(f'  E Acc: {e_acc:.2f}%, I Acc: {i_acc:.2f}%')
        
        if isinstance(optimizer, list):
            logger.info(f'  LR: {optimizer[0].param_groups[0]["lr"]:.2e}')
        else:
            logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model with improved logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model(s)
            if isinstance(model, list):
                for i, m in enumerate(model):
                    torch.save(m.state_dict(), f'best_advanced_model_{i}_{RUN_TIMESTAMP}.pth')
            else:
                torch.save(model.state_dict(), f'best_advanced_model_{RUN_TIMESTAMP}.pth')
            
            logger.info(f'  NEW BEST: {best_acc:.2f}% (saved)')
        else:
            patience_counter += 1
        
        # Early success check
        if val_acc >= 90.0:
            logger.info(f'SUCCESS! Reached 90% accuracy at epoch {epoch}')
            break
        
        # Early stopping check
        if patience_counter >= patience_limit and epoch > 50:
            logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)')
            break
        
        logger.info('-' * 60)
    
    # Final plot
    plot_learning_curves(train_losses, train_accs, val_losses, val_accs, 
                       learning_rates, e_accs, i_accs, RUN_TIMESTAMP)

    logger.info(f'Training complete!')
    logger.info(f'Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch})')
    
    # Final evaluation
    logger.info('\nFinal Classification Report:')
    logger.info(classification_report(val_targets, val_preds, target_names=['E', 'I']))
    
    logger.info('\nFinal Confusion Matrix:')
    logger.info(str(confusion_matrix(val_targets, val_preds)))

if __name__ == '__main__':
    main() 