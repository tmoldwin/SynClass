"""
2D CNN synapse classifier with multi-channel input (image + pre mask + post mask)
Uses modular components with multiple augmentations per epoch
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import modular components
from constants import setup_logging
from data_loader import prepare_synapse_data
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary, compute_class_weights


class MultiChannelSynapseDataset(torch.utils.data.Dataset):
    """2D dataset with 3 channels: image + pre_mask + post_mask, multiple augments per epoch."""
    
    def __init__(self, file_list, synapse_map, data_dir=None, augment=False, 
                 input_size=224, augments_per_epoch=3):
        from constants import DATA_DIR
        import os
        
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.augment = augment
        self.input_size = input_size
        self.augments_per_epoch = augments_per_epoch if augment else 1
        
        # Get 2D augmentation transforms
        from augmentation import get_2d_augmentation_transform
        self.transform = get_2d_augmentation_transform(augment=augment, input_size=input_size)
        
        print(f"📊 Dataset: {len(file_list)} synapses × {self.augments_per_epoch} augments = {len(self)} samples per epoch")
    
    def _rotated_crop(self, image, center_h, center_w, crop_size, angle):
        """Apply rotated crop to image without black borders."""
        import cv2
        
        h, w = image.shape
        
        # Create rotation matrix around crop center
        rotation_matrix = cv2.getRotationMatrix2D((float(center_w), float(center_h)), float(angle), 1.0)
        
        # Rotate the entire image
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Extract crop from rotated image
        start_h = center_h - crop_size // 2
        start_w = center_w - crop_size // 2
        end_h = start_h + crop_size
        end_w = start_w + crop_size
        
        # Ensure crop stays within bounds
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, end_h)
        end_w = min(w, end_w)
        
        cropped = rotated[start_h:end_h, start_w:end_w]
        
        # If crop is smaller than expected, pad with reflection
        if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
            pad_h = max(0, crop_size - cropped.shape[0])
            pad_w = max(0, crop_size - cropped.shape[1])
            cropped = cv2.copyMakeBorder(cropped, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        return cropped
        
    def __len__(self):
        return len(self.file_list) * self.augments_per_epoch
    
    def __getitem__(self, idx):
        import os
        import numpy as np
        from torchvision.transforms import functional as TF
        
        # Map idx back to original synapse and augmentation number
        synapse_idx = idx // self.augments_per_epoch
        aug_idx = idx % self.augments_per_epoch
        
        filename = self.file_list[synapse_idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # Load synapse data (3D)
        data_3d = np.load(filepath)
        
        # Get label
        syn_id = int(filename.split('_')[0])
        syn_type = self.synapse_map[syn_id]
        label = 1 if syn_type == 'I' else 0  # I=1 (inhibitory), E=0 (excitatory)
        
        # Load masks
        try:
            pre_mask_path = os.path.join(self.data_dir, filename.replace('syn.npy', 'pre_syn_n_mask.npy'))
            post_mask_path = os.path.join(self.data_dir, filename.replace('syn.npy', 'post_syn_n_mask.npy'))
            pre_mask_3d = np.load(pre_mask_path)
            post_mask_3d = np.load(post_mask_path)
        except FileNotFoundError:
            # Create dummy masks if not found
            pre_mask_3d = np.zeros_like(data_3d)
            post_mask_3d = np.zeros_like(data_3d)
        
        # Take middle slice for 2D processing
        mid_z = data_3d.shape[2] // 2
        data_2d = data_3d[:, :, mid_z]
        pre_mask_2d = pre_mask_3d[:, :, mid_z]
        post_mask_2d = post_mask_3d[:, :, mid_z]
        
        # Normalize image data 
        data_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-8)
        
        # Convert to uint8 for PIL conversion
        data_uint8 = (data_norm * 255).astype(np.uint8)
        pre_uint8 = (pre_mask_2d * 255).astype(np.uint8)
        post_uint8 = (post_mask_2d * 255).astype(np.uint8)
        
        # Smart augmentation with rotated crops
        if self.augment:
            # Set random seed for consistent augmentation
            seed = hash((synapse_idx, aug_idx)) % (2**31)
            np.random.seed(seed % (2**31))
            
            # Random vertical flip (keep this)
            if np.random.random() > 0.5:
                data_uint8 = np.flipud(data_uint8)
                pre_uint8 = np.flipud(pre_uint8)
                post_uint8 = np.flipud(post_uint8)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                data_uint8 = np.fliplr(data_uint8)
                pre_uint8 = np.fliplr(pre_uint8)
                post_uint8 = np.fliplr(post_uint8)
            
            # Rotated crop (65-80% of original size)
            h, w = data_uint8.shape
            crop_ratio = np.random.uniform(0.65, 0.80)
            crop_size = int(min(h, w) * crop_ratio)
            
            # Random rotation angle
            angle = np.random.uniform(-45, 45)
            
            # Find crop center (mask-aware if possible)
            combined_mask = (pre_mask_2d > 0) | (post_mask_2d > 0)
            if np.any(combined_mask):
                # Center around mask region
                mask_coords = np.where(combined_mask)
                center_h = int(np.mean(mask_coords[0]))
                center_w = int(np.mean(mask_coords[1]))
                
                # Add some randomness
                offset_h = np.random.randint(-h//8, h//8)
                offset_w = np.random.randint(-w//8, w//8)
                center_h = np.clip(center_h + offset_h, crop_size//2, h - crop_size//2)
                center_w = np.clip(center_w + offset_w, crop_size//2, w - crop_size//2)
            else:
                # Random center
                center_h = np.random.randint(crop_size//2, h - crop_size//2)
                center_w = np.random.randint(crop_size//2, w - crop_size//2)
            
            # Apply rotated crop to all channels
            data_uint8 = self._rotated_crop(data_uint8, center_h, center_w, crop_size, angle)
            pre_uint8 = self._rotated_crop(pre_uint8, center_h, center_w, crop_size, angle)
            post_uint8 = self._rotated_crop(post_uint8, center_h, center_w, crop_size, angle)
            
            # Random noise to data only (not masks)
            noise = np.random.normal(0, 0.03 * 255, data_uint8.shape).astype(np.int16)
            data_uint8 = np.clip(data_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Resize all channels to target size
        import cv2
        data_resized = cv2.resize(data_uint8, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        pre_resized = cv2.resize(pre_uint8, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        post_resized = cv2.resize(post_uint8, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors and normalize
        data_tensor = torch.tensor(data_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        pre_tensor = torch.tensor(pre_resized, dtype=torch.float32).unsqueeze(0) / 255.0  
        post_tensor = torch.tensor(post_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # Normalize data channel with ImageNet stats
        data_tensor = TF.normalize(data_tensor, mean=[0.485], std=[0.229])
        
        # Stack all channels: [image, pre_mask, post_mask]
        multi_channel = torch.cat([data_tensor, pre_tensor, post_tensor], dim=0)  # Shape: (3, H, W)
        
        return multi_channel, label


class CNN2DMultiChannel(nn.Module):
    """2D CNN for 3-channel input: image + pre_mask + post_mask."""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, cnn_depth=5):
        super().__init__()
        
        # Input: 3 channels (image + pre_mask + post_mask)
        if cnn_depth == 3:
            # Lighter 3-block architecture
            self.features = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third conv block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 256
        elif cnn_depth == 5:
            # Medium 5-block architecture
            self.features = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third conv block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fourth conv block with residual-like connection
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fifth conv block with global average pooling for regularization
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 512
        else:  # cnn_depth == 7
            # Deeper 7-block architecture
            self.features = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third conv block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fourth conv block
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fifth conv block
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Sixth conv block
                nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                
                # Seventh conv block
                nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 768
        
        # Classifier with better regularization (minimal dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Light dropout only here
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_multichannel_dataloaders(train_files, test_files, synapse_map, 
                                   batch_size=16, num_workers=4, pin_memory=True,
                                   input_size=224, augment_train=True, augments_per_epoch=3):
    """Create dataloaders for multi-channel training."""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MultiChannelSynapseDataset(
        train_files, synapse_map, augment=augment_train, 
        input_size=input_size, augments_per_epoch=augments_per_epoch
    )
    val_dataset = MultiChannelSynapseDataset(
        test_files, synapse_map, augment=False, 
        input_size=input_size, augments_per_epoch=1  # No multiple augments for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='2D CNN Multi-Channel Synapse Classifier')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--augments_per_epoch', type=int, default=3, help='Number of augmentations per synapse per epoch')
    parser.add_argument('--cnn_depth', type=int, default=5, help='Number of CNN blocks (3, 5, or 7)')
    parser.add_argument('--run_name', type=str, help='Run name for this experiment')
    
    args = parser.parse_args()
    
    # Setup
    model_name = args.run_name if args.run_name else '2dcnn_multichannel'
    logger = setup_logging(model_name)
    logger.info("Starting 2D CNN Multi-Channel Synapse Classification")
    
    set_random_seeds(42)
    device = get_device()
    
    # Load data
    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(logger=logger)
    
    # Create dataloaders
    train_loader, val_loader = create_multichannel_dataloaders(
        train_files, test_files, synapse_map,
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True,
        input_size=args.input_size,
        augment_train=True,
        augments_per_epoch=args.augments_per_epoch
    )
    
    # Initialize model
    model = CNN2DMultiChannel(num_classes=2, dropout_rate=args.dropout_rate, cnn_depth=args.cnn_depth).to(device)
    print_model_summary(model, logger=logger)
    
    # Setup training with enhanced regularization
    class_weights = compute_class_weights(train_files, synapse_map)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=args.lr/1000
    )
    
    # Train
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, device, scheduler, f'best_synapse_model_{model_name}.pth',
        early_stopping_patience=25, logger=logger
    )
    
    # Save plots
    save_all_plots(results, 'figures', model_name)
    
    logger.info(f"Training complete! Best accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")


if __name__ == '__main__':
    main()
