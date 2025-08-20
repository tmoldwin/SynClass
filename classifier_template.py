"""
Template for creating new synapse classifiers using modular components

This template shows how to create a new classifier with minimal boilerplate code.
Just define your model architecture and configure the training parameters.
"""
import argparse
import torch
import torch.nn as nn

# Import our modular components
from constants import setup_logging
from data_loader import prepare_synapse_data
from datasets import create_dataloaders
from training import train_model, setup_training
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary, log_system_info


class YourCustomModel(nn.Module):
    """Define your custom model architecture here."""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # Define your model layers here
        # Example for 3D CNN:
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension for 3D conv
        
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Custom synapse classifier')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=64, help='Input size')
    parser.add_argument('--use_3d', action='store_true', help='Use 3D data (default: 2D)')
    parser.add_argument('--save_plots', action='store_true', help='Save training plots')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('custom_classifier')
    logger.info("Starting Custom Synapse Classification Training")
    logger.info("=" * 50)
    
    # Log system information
    log_system_info(logger)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Log hyperparameters
    logger.info(f"Hyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Input Size: {args.input_size}")
    logger.info(f"  Dropout Rate: {args.dropout_rate}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  Use 3D: {args.use_3d}")
    
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
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        is_3d=args.use_3d,
        input_size=args.input_size,
        augment_train=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = YourCustomModel(
        num_classes=2,
        dropout_rate=args.dropout_rate
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
    
    # Train model
    logger.info("Starting training...")
    save_path = "best_custom_model.pth"
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
        early_stopping_patience=10,
        logger=logger,
        save_best=True
    )
    
    # Save plots if requested
    if args.save_plots:
        logger.info("Saving training plots...")
        save_all_plots(
            training_results=training_results,
            save_dir='figures',
            model_name='custom_classifier'
        )
    
    # Final summary
    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {training_results['best_val_accuracy']:.2f}%")
    logger.info(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
