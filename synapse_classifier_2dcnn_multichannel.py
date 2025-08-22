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
    """2D dataset with 3 channels: image + pre_mask + post_mask, configurable examples per epoch."""
    
    def __init__(self, file_list, synapse_map, data_dir=None, augment=False, 
                 input_size=224, examples_per_epoch=None, augment_prob=0.2):
        from constants import DATA_DIR
        import os
        
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.augment = augment
        self.input_size = input_size
        self.examples_per_epoch = examples_per_epoch if examples_per_epoch is not None else len(file_list)
        self.augment_prob = augment_prob  # Probability of applying augmentation vs using original
        
        # Get 2D augmentation transforms
        from augmentation import get_2d_augmentation_transform
        self.transform = get_2d_augmentation_transform(augment=augment, input_size=input_size)
        
        print(f"📊 Dataset: {len(file_list)} total synapses -> {self.examples_per_epoch} examples per epoch")
        if augment:
            print(f"🎲 Augmentation probability: {augment_prob:.1f} (vs {1-augment_prob:.1f} for originals)")
    
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
    
    def _create_balanced_epoch_indices(self):
        """Create balanced indices for sampling examples per epoch."""
        import numpy as np
        
        # Get labels for all files
        labels = []
        for filename in self.file_list:
            syn_id = int(filename.split('_')[0])
            syn_type = self.synapse_map[syn_id]
            labels.append(syn_type)
        
        # Separate indices by class
        e_indices = [i for i, label in enumerate(labels) if label == 'E']
        i_indices = [i for i, label in enumerate(labels) if label == 'I']
        
        # Calculate how many samples per class we need (balanced)
        samples_per_class = self.examples_per_epoch // 2
        
        # Sample balanced indices
        np.random.seed(42)  # Fixed seed for reproducibility
        selected_e = np.random.choice(e_indices, min(samples_per_class, len(e_indices)), replace=False)
        selected_i = np.random.choice(i_indices, min(samples_per_class, len(i_indices)), replace=False)
        
        # Combine and shuffle
        all_selected = np.concatenate([selected_e, selected_i])
        np.random.shuffle(all_selected)
        
        # Adjust final count to match examples_per_epoch exactly
        self._epoch_indices = all_selected[:self.examples_per_epoch]
        
        print(f"🎯 Balanced sampling: {len(selected_e)} E + {len(selected_i)} I = {len(self._epoch_indices)} total examples")
        
    def __len__(self):
        return self.examples_per_epoch
    
    def __getitem__(self, idx):
        import os
        import numpy as np
        from torchvision.transforms import functional as TF
        
        # Map idx to a synapse (with balanced sampling if examples_per_epoch < len(file_list))
        if self.examples_per_epoch >= len(self.file_list):
            # If we want more examples than synapses, repeat synapses with different augmentations
            synapse_idx = idx % len(self.file_list)
            aug_idx = idx // len(self.file_list)
        else:
            # If we want fewer examples, sample with balanced class distribution
            # Create balanced sample indices once per epoch (deterministic)
            if not hasattr(self, '_epoch_indices'):
                self._create_balanced_epoch_indices()
            synapse_idx = self._epoch_indices[idx]
            aug_idx = 0
        
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
    
    def __init__(self, num_classes=2, dropout_rate=0.3, cnn_depth=3):
        super().__init__()
        
        # Input: 3 channels (image + pre_mask + post_mask)
        if cnn_depth == 1:
            # Minimal 1-block architecture
            self.features = nn.Sequential(
                # Single conv block
                nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 128
        elif cnn_depth == 2:
            # Light 2-block architecture
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
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 128
        elif cnn_depth == 3:
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
            # Deep 5-block architecture
            self.features = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second conv block
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third conv block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fourth conv block
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fifth conv block
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 512
        else:
            raise ValueError(f"Unsupported cnn_depth: {cnn_depth}. Must be 1, 2, 3, or 5.")
        
        # Classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Use the actual dropout_rate parameter
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
                                   input_size=224, augment_train=True, examples_per_epoch=None,
                                   augment_val=False, val_examples_per_epoch=None):
    """Create dataloaders for multi-channel training."""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MultiChannelSynapseDataset(
        train_files, synapse_map, augment=augment_train, 
        input_size=input_size, examples_per_epoch=examples_per_epoch
    )
    val_dataset = MultiChannelSynapseDataset(
        test_files, synapse_map, augment=augment_val, 
        input_size=input_size, examples_per_epoch=val_examples_per_epoch
    )
    
    print(f"🎯 TRAINING CONFIGURATION:")
    print(f"   Training: {len(train_files)} total synapses -> {len(train_dataset)} examples per epoch")
    print(f"   Validation: {len(test_files)} total synapses -> {len(val_dataset)} examples per epoch")
    print(f"   Training/Val ratio: {len(train_dataset)/len(val_dataset):.1f}:1")
    
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--examples_per_epoch', type=int, default=None, help='Number of training examples per epoch (default: all)')
    parser.add_argument('--cnn_depth', type=int, default=3, help='Number of CNN blocks (1, 3, or 5)')
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
    
    # Verify balanced dataset - should be 50/50 E/I for BOTH train and test
    logger.info(f"📊 CLASS VERIFICATION:")
    logger.info(f"   Train: E={data_stats['train_distribution']['E']}, I={data_stats['train_distribution']['I']}")
    logger.info(f"   Test:  E={data_stats['test_distribution']['E']}, I={data_stats['test_distribution']['I']}")
    
    train_ratio = data_stats['train_distribution']['E'] / data_stats['train_distribution']['I']
    test_ratio = data_stats['test_distribution']['E'] / data_stats['test_distribution']['I']
    
    logger.info(f"   Train E/I ratio: {train_ratio:.3f}:1 (should be ~1.0)")
    logger.info(f"   Test E/I ratio:  {test_ratio:.3f}:1 (should be ~1.0)")
    
    train_balanced = abs(train_ratio - 1.0) <= 0.1
    test_balanced = abs(test_ratio - 1.0) <= 0.1
    
    if not train_balanced:
        logger.warning(f"⚠️  TRAIN dataset is NOT balanced! Ratio: {train_ratio:.3f}:1")
    if not test_balanced:
        logger.warning(f"⚠️  TEST dataset is NOT balanced! Ratio: {test_ratio:.3f}:1")
        
    if train_balanced and test_balanced:
        logger.info(f"✅ Both train and test datasets are balanced - no class weights needed")
    else:
        logger.error(f"❌ Dataset imbalance detected - this will cause training issues!")
    
    # Create dataloaders
    train_loader, val_loader = create_multichannel_dataloaders(
        train_files, test_files, synapse_map,
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True,
        input_size=args.input_size,
        augment_train=False,  # No augmentation 
        examples_per_epoch=args.examples_per_epoch,
        augment_val=False,  # No validation augmentation
        val_examples_per_epoch=None  # Use all validation examples
    )
    
    # Initialize model
    model = CNN2DMultiChannel(num_classes=2, dropout_rate=args.dropout_rate, cnn_depth=args.cnn_depth).to(device)
    print_model_summary(model, logger=logger)
    
    # Setup training - no class weights needed since dataset is balanced
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use exponential decay for more predictable LR reduction
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # 2% reduction per epoch
    
    # Train
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, device, scheduler, f'best_synapse_model_{model_name}.pth',
        early_stopping_patience=15, logger=logger
    )
    
    # Save plots
    save_all_plots(results, 'figures', model_name)
    
    logger.info(f"Training complete! Best accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")


if __name__ == '__main__':
    main()
