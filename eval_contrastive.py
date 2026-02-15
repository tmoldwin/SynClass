"""Evaluate a contrastive checkpoint without stopping the running training."""
import argparse
import os
import torch

from constants import setup_logging, DATA_DIR, DATA_ARCHIVE
from data_loader import prepare_synapse_data
from synapse_contrastive import (
    SimCLRModel, extract_embeddings, evaluate_representations,
    plot_contrastive_loss, plot_lr_curve, load_checkpoint, get_latest_checkpoint,
    CONTRASTIVE_EVAL_DIR, CONTRASTIVE_TRAINING_DIR,
)
from utils import get_device


def main():
    parser = argparse.ArgumentParser(description='Evaluate contrastive checkpoint')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help='Path to checkpoint or "latest"')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--cnn_depth', type=int, default=3)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--multichannel', action='store_true', default=False,
                        help='Use 3-channel model (default: EM-only)')
    args = parser.parse_args()
    em_only = not args.multichannel
    in_channels = 1 if em_only else 3

    logger = setup_logging('contrastive_eval')
    device = get_device(prefer_gpu=not args.cpu)

    # Find checkpoint
    ckpt_path = (get_latest_checkpoint() if args.checkpoint == 'latest'
                 else args.checkpoint)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        logger.error(f"No checkpoint found: {ckpt_path}")
        return
    logger.info(f"Evaluating checkpoint: {ckpt_path}")

    # Load model
    model = SimCLRModel(cnn_depth=args.cnn_depth, proj_dim=args.proj_dim,
                        in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # dummy, needed for load
    epoch, losses, lr_history = load_checkpoint(
        ckpt_path, model, optimizer, None, device, logger)

    # Plot training curves so far
    os.makedirs(CONTRASTIVE_TRAINING_DIR, exist_ok=True)
    os.makedirs(CONTRASTIVE_EVAL_DIR, exist_ok=True)
    if losses:
        plot_contrastive_loss(losses, CONTRASTIVE_TRAINING_DIR)
    if lr_history:
        plot_lr_curve(lr_history, CONTRASTIVE_TRAINING_DIR)

    # Load data
    train_files, test_files, synapse_map, _ = prepare_synapse_data(logger=logger)
    all_files = train_files + test_files
    if args.max_samples:
        all_files = all_files[:args.max_samples]

    # Extract embeddings and evaluate
    logger.info(f"Extracting embeddings for {len(all_files)} samples (epoch {epoch})...")
    embeddings, labels = extract_embeddings(
        model, all_files, synapse_map, args.input_size, device,
        em_only=em_only)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    results = evaluate_representations(embeddings, labels, logger, CONTRASTIVE_EVAL_DIR)
    logger.info(f"Epoch {epoch} evaluation complete.")


if __name__ == '__main__':
    main()
