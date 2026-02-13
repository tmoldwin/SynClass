"""
2D CNN synapse classifier with multi-channel input (image + pre mask + post mask)
Uses modular components with multiple augmentations per epoch
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from constants import setup_logging
from data_loader import prepare_synapse_data
from preprocessing import create_multichannel_dataloaders
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary


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
    logger.info("CLASS VERIFICATION:")
    logger.info(f"   Train: E={data_stats['train_distribution']['E']}, I={data_stats['train_distribution']['I']}")
    logger.info(f"   Test:  E={data_stats['test_distribution']['E']}, I={data_stats['test_distribution']['I']}")
    
    train_ratio = data_stats['train_distribution']['E'] / data_stats['train_distribution']['I']
    test_ratio = data_stats['test_distribution']['E'] / data_stats['test_distribution']['I']
    
    logger.info(f"   Train E/I ratio: {train_ratio:.3f}:1 (should be ~1.0)")
    logger.info(f"   Test E/I ratio:  {test_ratio:.3f}:1 (should be ~1.0)")
    
    train_balanced = abs(train_ratio - 1.0) <= 0.1
    test_balanced = abs(test_ratio - 1.0) <= 0.1
    
    if not train_balanced:
        logger.warning(f"TRAIN dataset is NOT balanced! Ratio: {train_ratio:.3f}:1")
    if not test_balanced:
        logger.warning(f"TEST dataset is NOT balanced! Ratio: {test_ratio:.3f}:1")
        
    if train_balanced and test_balanced:
        logger.info("Both train and test datasets are balanced - no class weights needed")
    else:
        logger.error("Dataset imbalance detected - this will cause training issues!")
    
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
    # Use aggressive plateau-based scheduler that responds to validation accuracy
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # maximize validation accuracy
        factor=0.3,           # reduce LR by 70% when plateau detected
        patience=5,           # wait 5 epochs before reducing
        threshold=0.01,       # minimum improvement to be considered progress (1%)
        min_lr=args.lr/1000   # don't go below 0.1% of original LR
    )
    
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
