"""
ResNet-based synapse classifier with N randomly sampled z-slices as channels.
Default: N channels (EM only, one per slice). Option --use_masks: 3*N channels (EM, pre, post per slice).
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from constants import setup_logging
from data_loader import prepare_synapse_data
from preprocessing import create_multislice_dataloaders
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary


def resnet18_multichannel(num_classes=2, in_channels=3, dropout_rate=0.5, freeze_mode='layer4'):
    """ResNet18 with ImageNet pretrained weights.
    
    freeze_mode:
      'all'    — freeze entire backbone, only train fc
      'layer4' — freeze conv1-layer3, train layer4 + fc  (~2.6M trainable)
      'none'   — train everything
    
    For in_channels==3, uses pretrained conv1 directly.
    For in_channels!=3, replaces conv1 and copies pretrained weights for first 3 channels.
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if in_channels != 3:
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                for i in range(3, in_channels):
                    backbone.conv1.weight[:, i] = old_conv.weight[:, i % 3]

    frozen_prefixes = {
        'all': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
        'layer4': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'],
        'none': [],
    }
    prefixes = frozen_prefixes.get(freeze_mode, [])
    for name, param in backbone.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False

    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(128, num_classes)
    )
    return backbone


def main():
    parser = argparse.ArgumentParser(description='ResNet18 Synapse Classifier (N z-slices as channels)')
    parser.add_argument('--epochs', type=int, default=70, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=224, help='Input spatial size')
    parser.add_argument('--n_slices', type=int, default=3, help='Number of z-slices per volume (N channels; or 3*N with --use_masks)')
    parser.add_argument('--use_masks', action='store_true', help='Add pre/post mask channels per slice (3*N channels)')
    parser.add_argument('--freeze_mode', type=str, default='layer4', choices=['all', 'layer4', 'none'],
                        help='Freeze mode: all=only fc, layer4=train layer4+fc, none=train all')
    parser.add_argument('--examples_per_epoch', type=int, default=None, help='Training examples per epoch (default: all)')
    parser.add_argument('--run_name', type=str, help='Run name')
    parser.add_argument('--csv_path', type=str, default=None, help='Synapse CSV (id_, pre_clf_type)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory with *_syn.npy')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    model_name = args.run_name if args.run_name else f'resnet_multislice_n{args.n_slices}' + ('_masks' if args.use_masks else '')
    logger = setup_logging(model_name)
    ch = 3 * args.n_slices if args.use_masks else args.n_slices
    logger.info("ResNet18 multi-slice: n_slices=%s use_masks=%s -> %s channels, freeze_mode=%s, pretrained=ImageNet",
                args.n_slices, args.use_masks, ch, args.freeze_mode)
    set_random_seeds(42)
    device = get_device(prefer_gpu=not args.cpu)

    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(
        data_dir=args.data_dir, csv_path=args.csv_path, logger=logger)
    logger.info("CLASS VERIFICATION: Train E=%s I=%s | Test E=%s I=%s",
                data_stats['train_distribution']['E'], data_stats['train_distribution']['I'],
                data_stats['test_distribution']['E'], data_stats['test_distribution']['I'])

    in_channels = 3 * args.n_slices if args.use_masks else args.n_slices
    train_loader, val_loader = create_multislice_dataloaders(
        train_files, test_files, synapse_map, n_slices=args.n_slices, use_masks=args.use_masks,
        batch_size=args.batch_size, num_workers=0, pin_memory=False,
        input_size=args.input_size, augment_train=True, examples_per_epoch=args.examples_per_epoch,
        augment_val=False, val_examples_per_epoch=None, data_dir=args.data_dir
    )

    model = resnet18_multichannel(
        num_classes=2, in_channels=in_channels, dropout_rate=args.dropout_rate,
        freeze_mode=args.freeze_mode
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s trainable / %s total (%.1f%% frozen)",
                f'{trainable:,}', f'{total:,}', 100 * (1 - trainable / total))
    print_model_summary(model, logger=logger)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=5, threshold=0.01, min_lr=args.lr / 1000
    )

    results = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, device, scheduler, f'best_synapse_model_{model_name}.pth',
        early_stopping_patience=7, logger=logger
    )
    save_all_plots(results, 'figures', model_name)
    logger.info("Training complete! Best val accuracy: %.2f%%", results.get('best_val_accuracy', 0))


if __name__ == '__main__':
    main()
