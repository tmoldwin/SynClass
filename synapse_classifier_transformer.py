"""
Pure transformer synapse classifier - no convolutions.
Uses patch embedding, transformer encoder blocks with residuals, and classification head.
"""
import argparse
import math
import torch
import torch.nn as nn

from constants import setup_logging
from data_loader import prepare_synapse_data
from preprocessing import create_multichannel_dataloaders
from training import train_model
from plotting import save_all_plots
from utils import set_random_seeds, get_device, print_model_summary


class PatchEmbed(nn.Module):
    """Split image into patches and linearly embed. No convolutions."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == W == self.img_size
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, self.num_patches, p * p * C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block: self-attention + FFN, both with residuals."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self._attn_block(self.norm1(x))
        x = x + self._mlp_block(self.norm2(x))
        return x

    def _attn_block(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return attn_out

    def _mlp_block(self, x):
        return self.mlp(x)


class SynapseTransformer(nn.Module):
    """Pure transformer for synapse classification. No convolutions."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description='Pure Transformer Synapse Classifier')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--examples_per_epoch', type=int, default=None)
    parser.add_argument('--run_name', type=str, default='transformer')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (e.g. for incompatible GPU like RTX 5060)')

    args = parser.parse_args()

    model_name = args.run_name
    logger = setup_logging(model_name)
    logger.info("Starting Pure Transformer Synapse Classification")

    set_random_seeds(42)
    device = get_device(prefer_gpu=not args.cpu)

    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(logger=logger)
    logger.info(f"Train: E={data_stats['train_distribution']['E']}, I={data_stats['train_distribution']['I']}")
    logger.info(f"Test:  E={data_stats['test_distribution']['E']}, I={data_stats['test_distribution']['I']}")

    train_loader, val_loader = create_multichannel_dataloaders(
        train_files, test_files, synapse_map,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        input_size=args.input_size,
        augment_train=False,
        examples_per_epoch=args.examples_per_epoch,
        augment_val=False,
        val_examples_per_epoch=None,
    )

    model = SynapseTransformer(
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    print_model_summary(model, logger=logger)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=5, threshold=0.01, min_lr=args.lr / 1000
    )

    results = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, device, scheduler, f'best_synapse_model_{model_name}.pth',
        early_stopping_patience=15, logger=logger
    )

    save_all_plots(results, 'figures', model_name)
    logger.info(f"Training complete! Best accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")


if __name__ == '__main__':
    main()
