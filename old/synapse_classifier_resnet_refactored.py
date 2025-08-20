"""
Refactored ResNet-based synapse classifier using modular components
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Import our modular components
from constants import setup_logging
from data_loader import prepare_synapse_data
from datasets import create_dataloaders
from training import train_model, setup_training
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary, log_system_info


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ResNetClassifier(nn.Module):
    """ResNet-based classifier with attention mechanism for synapse classification."""
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3, resnet_version='resnet152'):
        super().__init__()
        
        # Load ResNet backbone
        if resnet_version == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        elif resnet_version == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet_version == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        num_ftrs = self.resnet.fc.in_features
        
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs // 4, num_ftrs),
            nn.Sigmoid()
        )
        
        # Large classifier head (optimized based on sweep analysis)
        self.classifier = nn.Sequential(
            # First layer: 2048 -> 1024
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
            nn.Dropout(dropout_rate * 0.5),
            
            # Fourth layer: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            
            # Final layer: 128 -> num_classes
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Extract features with ResNet backbone
        features = self.resnet(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        return self.classifier(attended_features)


def optimize_batch_size_for_gpu(device, default_batch_size=16):
    """Optimize batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return default_batch_size
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f'GPU: {gpu_name}')
    print(f'GPU Memory: {gpu_memory:.1f} GB')
    
    # Optimize batch size based on GPU memory
    if gpu_memory >= 24:  # High-end GPU (RTX 4090, A100, etc.)
        batch_size = 64
        print(f'High-end GPU detected, using batch size: {batch_size}')
    elif gpu_memory >= 12:  # Mid-range GPU (RTX 3080, etc.)
        batch_size = 32
        print(f'Mid-range GPU detected, using batch size: {batch_size}')
    elif gpu_memory >= 8:  # Lower-end GPU
        batch_size = 16
        print(f'Lower-end GPU detected, using batch size: {batch_size}')
    else:  # Very limited GPU memory
        batch_size = 8
        print(f'Limited GPU memory, using batch size: {batch_size}')
    
    return batch_size


def get_num_workers_for_gpu():
    """Get optimal number of workers based on GPU type."""
    if not torch.cuda.is_available():
        return 4
    
    gpu_name = torch.cuda.get_device_name(0)
    
    if 'A100' in gpu_name or 'H100' in gpu_name:
        num_workers = 8
        print(f'High-end GPU detected, using {num_workers} workers')
    elif 'RTX' in gpu_name or 'V100' in gpu_name:
        num_workers = 6
        print(f'Mid-range GPU detected, using {num_workers} workers')
    else:
        num_workers = 4
        print(f'Using {num_workers} workers')
    
    return num_workers


def main():
    """Main training function for ResNet synapse classifier."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='ResNet-based synapse classifier')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto if None)')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss')
    parser.add_argument('--resnet_version', type=str, default='resnet152', 
                       choices=['resnet50', 'resnet101', 'resnet152'], help='ResNet version')
    parser.add_argument('--save_plots', action='store_true', help='Save training plots')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('resnet_refactored')
    logger.info("Starting ResNet Synapse Classification Training (Refactored)")
    logger.info("=" * 60)
    
    # Log system information
    log_system_info(logger)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Get device and optimize settings
    device = get_device()
    
    # Optimize batch size and workers based on GPU
    if args.batch_size is None:
        batch_size = optimize_batch_size_for_gpu(device)
    else:
        batch_size = args.batch_size
    
    num_workers = get_num_workers_for_gpu()
    
    # Log hyperparameters
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Input Size: {args.input_size}")
    logger.info(f"  Dropout Rate: {args.dropout_rate}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  Label Smoothing: {args.label_smoothing}")
    logger.info(f"  ResNet Version: {args.resnet_version}")
    logger.info(f"  Use Focal Loss: {args.use_focal_loss}")
    
    # Prepare data
    logger.info("Preparing synapse data...")
    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(
        test_size=0.2, random_seed=42, logger=logger
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_files=train_files,
        test_files=test_files,
        synapse_map=synapse_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        is_3d=False,  # ResNet uses 2D data
        input_size=args.input_size,
        augment_train=True
    )
    
    # Initialize model
    logger.info(f"Initializing {args.resnet_version} model...")
    model = ResNetClassifier(
        num_classes=2,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        resnet_version=args.resnet_version
    ).to(device)
    
    # Print model summary
    print_model_summary(model, logger=logger)
    
    # Setup training components
    logger.info("Setting up training components...")
    criterion, optimizer, scheduler = setup_training(
        model=model,
        train_files=train_files,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Override with focal loss if requested
    if args.use_focal_loss:
        # Get class weights for focal loss alpha
        from utils import compute_class_weights
        class_weights = compute_class_weights(train_files, synapse_map)
        criterion = FocalLoss(alpha=class_weights[1], gamma=2.0)
        logger.info("Using Focal Loss instead of CrossEntropyLoss")
    else:
        # Add label smoothing to CrossEntropyLoss
        from utils import compute_class_weights
        class_weights = compute_class_weights(train_files, synapse_map)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        logger.info(f"Using CrossEntropyLoss with label smoothing: {args.label_smoothing}")
    
    # Use enhanced optimizer and scheduler (based on original improvements)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=args.lr/100
    )
    
    # Resume from checkpoint if requested
    save_path = f"best_synapse_model_resnet_{args.resnet_version.lower()}.pth"
    if args.resume:
        from training import resume_training
        if resume_training(model, save_path, device):
            logger.info(f"Resumed training from {save_path}")
        else:
            logger.info("Could not resume training, starting from scratch")
    
    # Train model
    logger.info("Starting training...")
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        save_path=save_path,
        early_stopping_patience=15,
        logger=logger,
        save_best=True
    )
    
    # Save plots if requested
    if args.save_plots:
        logger.info("Saving training plots...")
        save_all_plots(
            training_results=training_results,
            save_dir='figures',
            model_name=f'resnet_{args.resnet_version.lower()}'
        )
    
    # Final summary
    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
    logger.info(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
