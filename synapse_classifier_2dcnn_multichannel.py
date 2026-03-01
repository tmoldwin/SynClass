"""
2D CNN synapse classifier with multi-channel input (image + pre mask + post mask)
Uses modular components with multiple augmentations per epoch.
Supports optional transformer block on conv features (conv-stem + a bit of transformer).
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

from constants import setup_logging
from data_loader import prepare_synapse_data
from preprocessing import create_multichannel_dataloaders
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary


class ResidualBlock(nn.Module):
    """Residual block: conv-bn-relu, conv-bn, then + shortcut, relu."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class TransformerBlock(nn.Module):
    """Single transformer encoder block: self-attn + FFN, with pre-norm."""
    def __init__(self, dim, num_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or min(4 * dim, 512)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x + self._attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def _attn(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return attn_out


class CNN2DMultiChannel(nn.Module):
    """2D CNN for 2-channel input: EM*pre_mask + EM*post_mask (masked images)."""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, cnn_depth=3, use_res=True, use_transformer=False, transformer_layers=2, transformer_heads=4):
        super().__init__()
        self.use_transformer = use_transformer
        in_ch = 2
        if cnn_depth == 1:
            # Minimal 1-block architecture
            self.features = nn.Sequential(
                # Single conv block
                nn.Conv2d(in_ch, 128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            final_features = 128
        elif cnn_depth == 2:
            # Light 2-block architecture
            self.features = nn.Sequential(
                # First conv block
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
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
            if use_res:
                # Stem then 3 residual blocks (no final pool; done in forward for transformer support)
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Dropout2d(0.03),
                    ResidualBlock(64, 64),
                    nn.Dropout2d(0.03),
                    ResidualBlock(64, 128, stride=2),
                    nn.Dropout2d(0.05),
                    ResidualBlock(128, 256, stride=2),
                )
                final_features = 256
            else:
                # Lighter 3-block architecture (original) — ends with global pool
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                final_features = 256
        elif cnn_depth == 4:
            if use_res:
                # Bigger 4-block res backbone: 64->128->256->384
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Dropout2d(0.03),
                    ResidualBlock(64, 64),
                    nn.Dropout2d(0.03),
                    ResidualBlock(64, 128, stride=2),
                    nn.Dropout2d(0.05),
                    ResidualBlock(128, 256, stride=2),
                    nn.Dropout2d(0.05),
                    ResidualBlock(256, 384, stride=2),
                )
                final_features = 384
            else:
                raise ValueError("cnn_depth=4 requires use_res=True")
        elif cnn_depth == 5:
            if use_res:
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 32, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Dropout2d(0.03),
                    ResidualBlock(32, 64, stride=2),
                    nn.Dropout2d(0.03),
                    ResidualBlock(64, 128, stride=2),
                    nn.Dropout2d(0.05),
                    ResidualBlock(128, 256, stride=2),
                    nn.Dropout2d(0.05),
                    ResidualBlock(256, 512, stride=2),
                )
                final_features = 512
            else:
                # Deep 5-block architecture (original) — ends with global pool
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 32, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                final_features = 512
        else:
            raise ValueError(f"Unsupported cnn_depth: {cnn_depth}. Must be 1, 2, 3, 4, or 5.")
        
        # Res backbones output spatial map; depth 1/2 and non-res output 1x1
        self.has_spatial_output = use_res and cnn_depth in (3, 4, 5)
        
        if use_transformer:
            if not self.has_spatial_output:
                raise ValueError("use_transformer requires res backbone (cnn_depth 3, 4, or 5 with use_res)")
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(final_features, num_heads=transformer_heads, dropout=dropout_rate * 0.5)
                for _ in range(transformer_layers)
            ])
        else:
            self.transformer_layers = None
        
        # Classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        if self.has_spatial_output:
            # B, C, H, W
            if self.use_transformer:
                x = nn.functional.adaptive_avg_pool2d(x, (7, 7))  # B, C, 7, 7
                x = x.flatten(2).transpose(1, 2)   # B, 49, C
                for blk in self.transformer_layers:
                    x = blk(x)
                x = x.mean(dim=1)   # B, C
            else:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = x.flatten(1)
        else:
            x = x.flatten(1)
        x = self.classifier(x)
        return x


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='2D CNN Multi-Channel Synapse Classifier')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate (lower for stable val curves)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.35, help='Dropout rate (was 0.45; reduced for less aggressive regularization)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--examples_per_epoch', type=int, default=None, help='Number of training examples per epoch (default: all)')
    parser.add_argument('--cnn_depth', type=int, default=4, help='Number of CNN blocks (1, 2, 3, 4, or 5). Default 4 = bigger backbone.')
    parser.add_argument('--no_res', action='store_true', help='Disable residual connections (use plain CNN)')
    parser.add_argument('--no_transformer', action='store_true', help='Disable transformer block on conv features (default: use 2-layer transformer when res backbone)')
    parser.add_argument('--run_name', type=str, help='Run name for this experiment')
    parser.add_argument('--csv_path', type=str, default='Data/proofread_synapses/synapse_data.csv', help='Synapse CSV (id_, pre_clf_type). Use Data/proofread_synapses/synapse_data.csv for MICrONS proofread pipeline.')
    parser.add_argument('--data_dir', type=str, default='Data/proofread_synapses', help='Directory with *_syn.npy. Use Data/proofread_synapses for proofread data (no 7z).')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (e.g. for incompatible GPU like RTX 5060)')
    parser.add_argument('--resume', action='store_true', help='Load checkpoint and continue (warm start)')
    parser.add_argument('--low_priority', action='store_true', help='Run at below-normal OS priority to avoid freezing')
    
    args = parser.parse_args()
    
    # Setup
    model_name = args.run_name if args.run_name else '2dcnn_multichannel'
    logger = setup_logging(model_name)
    logger.info("Starting 2D CNN Multi-Channel Synapse Classification")
    
    set_random_seeds(42)
    
    if args.low_priority:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Running at below-normal priority")
    
    device = get_device(prefer_gpu=not args.cpu)
    
    # Load data
    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(
        data_dir=args.data_dir, csv_path=args.csv_path, logger=logger)
    
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
        num_workers=2,
        pin_memory=True,
        input_size=args.input_size,
        augment_train=True,   # Flips, rotated crop, noise for better generalization
        examples_per_epoch=args.examples_per_epoch,
        augment_val=False,    # No validation augmentation
        val_examples_per_epoch=None,  # Use all validation examples
        data_dir=args.data_dir
    )
    
    # Initialize model
    model = CNN2DMultiChannel(
        num_classes=2,
        dropout_rate=args.dropout_rate,
        cnn_depth=args.cnn_depth,
        use_res=not args.no_res,
        use_transformer=not args.no_transformer,
        transformer_layers=2,
        transformer_heads=4,
    ).to(device)
    save_path = f'best_synapse_model_{model_name}.pth'
    if args.resume and os.path.exists(save_path):
        from training import resume_training
        if resume_training(model, save_path, device):
            logger.info(f"Resumed from {save_path} (warm start)")
        else:
            logger.warning("Resume failed, starting from scratch")
    print_model_summary(model, logger=logger)
    
    # Setup training - no class weights needed since dataset is balanced
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
    from datetime import datetime
    figures_dir = os.path.join('figures', f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(figures_dir, exist_ok=True)
    
    results = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, device, scheduler, save_path,
        early_stopping_patience=15, logger=logger,
        figures_dir=figures_dir
    )
    
    # Save plots
    save_all_plots(results, figures_dir, model_name)
    
    logger.info(f"Training complete! Best accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")


if __name__ == '__main__':
    main()
