"""
ResNet-based synapse classifier using modular components
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Import modular components
from constants import setup_logging
from data_loader import prepare_synapse_data
from datasets import create_dataloaders
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary, compute_class_weights


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
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
    """ResNet-based classifier with attention mechanism."""
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super().__init__()
        
        # ResNet152 backbone
        self.resnet = models.resnet152(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs // 4, num_ftrs),
            nn.Sigmoid()
        )
        
        # Large classifier head (optimized from sweep analysis)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.resnet(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        return self.classifier(attended_features)


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='ResNet synapse classifier')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging('resnet')
    logger.info("Starting ResNet Synapse Classification")
    
    set_random_seeds(42)
    device = get_device()
    
    # Prepare data
    train_files, test_files, synapse_map, _ = prepare_synapse_data(logger=logger)
    train_loader, val_loader = create_dataloaders(
        train_files, test_files, synapse_map, 
        batch_size=args.batch_size, is_3d=False, input_size=args.input_size
    )
    
    # Initialize model
    model = ResNetClassifier(dropout_rate=args.dropout_rate).to(device)
    print_model_summary(model, logger=logger)
    
    # Setup training
    class_weights = compute_class_weights(train_files, synapse_map)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights[1], gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=args.lr/100
    )
    
    # Train
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        args.epochs, device, scheduler, 'best_synapse_model_resnet.pth',
        early_stopping_patience=15, logger=logger
    )
    
    # Save plots
    save_all_plots(results, 'figures', 'resnet')
    
    logger.info(f"Training complete! Best accuracy: {results['best_val_accuracy']:.2f}%")


if __name__ == '__main__':
    main()
