"""
SimCLR-style contrastive learning for synapse classification.
Learns representations without labels, then evaluates if E/I clusters emerge naturally.
Diagnoses whether labels or representations are the bottleneck.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from constants import setup_logging, DATA_DIR, DATA_ARCHIVE
from data_loader import prepare_synapse_data
from preprocessing import load_synapse_data, rotated_crop, preprocess_synapse_2d
from utils import set_random_seeds, get_device, print_model_summary

# All contrastive-related figures go here
CONTRASTIVE_FIG_DIR = os.path.join('figures', 'contrastive')
CONTRASTIVE_DIAGRAMS_DIR = os.path.join(CONTRASTIVE_FIG_DIR, 'diagrams')
CONTRASTIVE_TRAINING_DIR = os.path.join(CONTRASTIVE_FIG_DIR, 'training')
CONTRASTIVE_EVAL_DIR = os.path.join(CONTRASTIVE_FIG_DIR, 'evaluation')
CONTRASTIVE_DATA_DIR = os.path.join(CONTRASTIVE_FIG_DIR, 'data')


# ── Contrastive Augmentation ─────────────────────────────────────────────────

def contrastive_augment(data_3d, pre_mask_3d, post_mask_3d, input_size=224,
                        seed=None, em_only=True, z_index=None):
    """Strong augmentation for contrastive learning. Returns one augmented view.

    If z_index is None, uses middle slice (original behavior).
    If z_index is given, uses that Z-slice (for multi-slice: same synapse, different depth).
    If em_only=True, returns (1, H, W) tensor with just the EM channel.
    If em_only=False, returns (3, H, W) with EM + pre mask + post mask.
    """
    import cv2
    from torchvision.transforms import functional as TF

    rng = np.random.RandomState(seed)
    nz = data_3d.shape[2]

    if z_index is None:
        z_index = nz // 2
    else:
        z_index = max(0, min(int(z_index), nz - 1))

    data_2d = data_3d[:, :, z_index].copy()

    # Normalize EM to [0, 255]
    dmin, dmax = data_2d.min(), data_2d.max()
    data_uint8 = ((data_2d - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)

    if not em_only:
        pre_2d = pre_mask_3d[:, :, z_index].copy()
        post_2d = post_mask_3d[:, :, z_index].copy()
        pre_uint8 = (pre_2d * 255).astype(np.uint8) if pre_2d.max() <= 1.0 else pre_2d.astype(np.uint8)
        post_uint8 = (post_2d * 255).astype(np.uint8) if post_2d.max() <= 1.0 else post_2d.astype(np.uint8)

    # 1. Random flips
    if rng.random() > 0.5:
        data_uint8 = np.flipud(data_uint8).copy()
        if not em_only:
            pre_uint8 = np.flipud(pre_uint8).copy()
            post_uint8 = np.flipud(post_uint8).copy()
    if rng.random() > 0.5:
        data_uint8 = np.fliplr(data_uint8).copy()
        if not em_only:
            pre_uint8 = np.fliplr(pre_uint8).copy()
            post_uint8 = np.fliplr(post_uint8).copy()

    # 2. Random resized crop with rotation (scale 0.5-1.0, stronger than supervised)
    h, w = data_uint8.shape
    crop_ratio = rng.uniform(0.5, 1.0)
    crop_size = int(min(h, w) * crop_ratio)
    angle = rng.uniform(-45, 45)
    margin = crop_size // 2
    center_h = rng.randint(margin, max(margin + 1, h - margin))
    center_w = rng.randint(margin, max(margin + 1, w - margin))

    data_uint8 = rotated_crop(data_uint8, center_h, center_w, crop_size, angle)
    if not em_only:
        pre_uint8 = rotated_crop(pre_uint8, center_h, center_w, crop_size, angle)
        post_uint8 = rotated_crop(post_uint8, center_h, center_w, crop_size, angle)

    # 3. Intensity jitter (brightness + contrast)
    brightness = rng.uniform(0.7, 1.3)
    contrast = rng.uniform(0.7, 1.3)
    data_f = data_uint8.astype(np.float32)
    mean_val = data_f.mean()
    data_f = (data_f - mean_val) * contrast + mean_val * brightness
    data_uint8 = np.clip(data_f, 0, 255).astype(np.uint8)

    # 4. Gaussian blur (50% chance)
    if rng.random() > 0.5:
        ksize = int(rng.choice([3, 5]))
        data_uint8 = cv2.GaussianBlur(data_uint8, (ksize, ksize), 0)

    # 5. Gaussian noise
    noise = rng.normal(0, 0.05 * 255, data_uint8.shape).astype(np.int16)
    data_uint8 = np.clip(data_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Resize
    data_resized = cv2.resize(data_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)

    # To tensor + normalise
    data_t = torch.tensor(data_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    data_t = TF.normalize(data_t, mean=[0.485], std=[0.229])

    if em_only:
        return data_t  # (1, H, W)

    pre_resized = cv2.resize(pre_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    post_resized = cv2.resize(post_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    pre_t = torch.tensor(pre_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    post_t = torch.tensor(post_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    return torch.cat([data_t, pre_t, post_t], dim=0)  # (3, H, W)


# ── Dataset ──────────────────────────────────────────────────────────────────

class ContrastiveSynapseDataset(Dataset):
    """Returns two views of each synapse (+ label for eval).

    If use_multi_slice=True, the two views are different Z-slices from the same
    synapse (same identity, different depth). Otherwise both views are the
    middle slice with different random augmentations.
    """

    def __init__(self, file_list, synapse_map, input_size=224,
                 data_dir=None, archive_path=None, em_only=True,
                 use_multi_slice=False):
        from constants import DATA_DIR, DATA_ARCHIVE
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.input_size = input_size
        self.em_only = em_only
        self.use_multi_slice = use_multi_slice
        self.data_dir = data_dir or DATA_DIR
        self.archive_path = (
            archive_path if archive_path and os.path.isfile(archive_path)
            else (DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None)
        )
        ch = 1 if em_only else 3
        mode = "multi_slice" if use_multi_slice else "augment"
        print(f"ContrastiveDataset: {len(file_list)} synapses, {ch}ch, {mode}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        for attempt in range(3):
            try:
                data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
                    self.data_dir, self.archive_path, filename)
                break
            except (FileNotFoundError, EOFError, OSError):
                if attempt < 2:
                    idx = (idx + 1) % len(self.file_list)
                    filename = self.file_list[idx]
                else:
                    raise

        seed1 = int.from_bytes(os.urandom(4), 'big')
        seed2 = int.from_bytes(os.urandom(4), 'big')

        if self.use_multi_slice:
            nz = data_3d.shape[2]
            if nz >= 2:
                z1, z2 = np.random.choice(nz, size=2, replace=False)
            else:
                z1 = z2 = 0
            view1 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                        self.input_size, seed=seed1,
                                        em_only=self.em_only, z_index=z1)
            view2 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                        self.input_size, seed=seed2,
                                        em_only=self.em_only, z_index=z2)
        else:
            view1 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                        self.input_size, seed=seed1,
                                        em_only=self.em_only)
            view2 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                        self.input_size, seed=seed2,
                                        em_only=self.em_only)

        syn_id = int(filename.split('_')[0])
        label = 1 if self.synapse_map[syn_id] == 'I' else 0
        return view1, view2, label


# ── NT-Xent Loss ─────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (SimCLR)."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: (B, D) L2-normalised projection vectors."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                       # (2B, D)
        sim = F.cosine_similarity(z.unsqueeze(1),
                                  z.unsqueeze(0), dim=2)      # (2B, 2B)
        sim = sim / self.temperature

        # Mask out self-similarity on the diagonal
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, -1e9)

        # Positive pair targets: i <-> i+B
        targets = torch.cat([torch.arange(B, 2 * B),
                             torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, targets)


# ── SimCLR Model ─────────────────────────────────────────────────────────────

class SimCLRModel(nn.Module):
    """CNN encoder (from CNN2DMultiChannel) + MLP projection head."""

    def __init__(self, cnn_depth=3, proj_dim=64, hidden_dim=256, in_channels=1):
        super().__init__()
        from synapse_classifier_2dcnn_multichannel import CNN2DMultiChannel
        backbone = CNN2DMultiChannel(num_classes=2, dropout_rate=0.0,
                                     cnn_depth=cnn_depth)
        self.encoder = backbone.features

        # Patch first conv layer if not using 3 channels
        if in_channels != 3:
            old_conv = self.encoder[0]
            self.encoder[0] = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding)

        encoder_dim = {1: 128, 2: 128, 3: 256, 5: 512}[cnn_depth]
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def encode(self, x):
        """Encoder features (for downstream evaluation)."""
        h = self.encoder(x)
        return h.view(h.size(0), -1)

    def forward(self, x):
        """L2-normalised projection (for contrastive loss)."""
        h = self.encode(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


# ── Contrastive Training Loop ────────────────────────────────────────────────

# ── Checkpointing ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.path.join('saved_models', 'contrastive_checkpoints')


def save_checkpoint(path, epoch, model, optimizer, scheduler, losses, lr_history):
    """Save a training checkpoint that can be resumed later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'losses': losses,
        'lr_history': lr_history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device, logger):
    """Load a checkpoint and restore training state. Returns (start_epoch, losses, lr_history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    losses = ckpt.get('losses', [])
    lr_history = ckpt.get('lr_history', [])
    logger.info(f"Resumed from checkpoint: epoch {epoch}, loss {losses[-1]:.4f}" if losses
                else f"Resumed from checkpoint: epoch {epoch}")
    return epoch, losses, lr_history


def get_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """Find the most recent checkpoint file in the directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    return os.path.join(checkpoint_dir, ckpts[-1]) if ckpts else None


# ── Contrastive Training Loop ────────────────────────────────────────────────

def compute_contrastive_accuracy(model, dataloader, criterion, device, max_batches=5):
    """Compute how often the model correctly identifies its positive pair.

    Uses only the first max_batches batches for speed (avoids a full second pass).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (view1, view2, _) in enumerate(dataloader):
            if b >= max_batches:
                break
            view1, view2 = view1.to(device), view2.to(device)
            z1 = model(view1)
            z2 = model(view2)
            B = z1.size(0)
            z = torch.cat([z1, z2], dim=0)  # (2B, D)
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
            sim = sim / criterion.temperature
            mask = ~torch.eye(2 * B, dtype=torch.bool, device=device)
            sim = sim.masked_fill(~mask, -1e9)
            targets = torch.cat([torch.arange(B, 2 * B),
                                 torch.arange(0, B)]).to(device)
            preds = sim.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += 2 * B
    return correct / max(total, 1)


def train_contrastive(model, dataloader, optimizer, scheduler, criterion,
                      epochs, device, logger,
                      start_epoch=0, prev_losses=None, prev_lr_history=None,
                      checkpoint_every=10,
                      eval_fn=None):
    """Train encoder with NT-Xent. Returns loss history.

    Args:
        start_epoch: epoch to resume from (0-indexed, 0 = fresh start).
        prev_losses: loss history from prior training (for seamless resume).
        prev_lr_history: LR history from prior training.
        checkpoint_every: save a checkpoint every N epochs (0 = disabled).
        eval_fn: optional callable(model, epoch, losses, lr_history, snapshot)
            run at end of each epoch to evaluate and plot progress.
    """
    model.train()
    losses = list(prev_losses) if prev_losses else []
    lr_history = list(prev_lr_history) if prev_lr_history else []
    contrastive_accs = []
    total_batches = len(dataloader)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        n_batches = 0
        log_interval = max(1, total_batches // 5)
        snapshot = None

        for view1, view2, batch_labels in dataloader:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Capture first batch every epoch for visualization
            if snapshot is None:
                with torch.no_grad():
                    h1 = model.encode(view1).cpu()
                    h2 = model.encode(view2).cpu()
                    z1_snap = model(view1).cpu()
                    z2_snap = model(view2).cpu()
                    z_cat = torch.cat([z1_snap, z2_snap], dim=0)
                    sim = F.cosine_similarity(
                        z_cat.unsqueeze(1), z_cat.unsqueeze(0), dim=2)
                    sim = sim / criterion.temperature
                    snapshot = {
                        'sim_matrix': sim.numpy(),
                        'z1': z1_snap.numpy(), 'z2': z2_snap.numpy(),
                        'h1': h1.numpy(), 'h2': h2.numpy(),
                        'labels': batch_labels.numpy(),
                    }

            if n_batches % log_interval == 0 or n_batches == total_batches:
                running_avg = epoch_loss / n_batches
                logger.info(f"  [{n_batches:4d}/{total_batches}] loss: {loss.item():.4f}  avg: {running_avg:.4f}")
                sys.stdout.flush()

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if scheduler is not None:
            scheduler.step(avg_loss)

        lr_now = optimizer.param_groups[0]['lr']
        lr_history.append(lr_now)

        # Contrastive accuracy (positive pair identification)
        contra_acc = compute_contrastive_accuracy(model, dataloader, criterion, device)
        contrastive_accs.append(contra_acc)

        logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                     f"Loss {avg_loss:.4f} | LR {lr_now:.2e} | "
                     f"Contrastive acc {contra_acc:.1%}")

        # Periodic checkpoint
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch{epoch + 1}.pth')
            save_checkpoint(ckpt_path, epoch + 1, model, optimizer, scheduler,
                            losses, lr_history)
            logger.info(f"Checkpoint saved -> {ckpt_path}")

        # Per-epoch evaluation & plots (pass snapshot for viz)
        if eval_fn is not None:
            eval_fn(model, epoch + 1, losses, lr_history, snapshot,
                    contrastive_accs)
            model.train()

    # Always save a final checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch{epochs}.pth')
    save_checkpoint(ckpt_path, epochs, model, optimizer, scheduler,
                    losses, lr_history)
    logger.info(f"Final checkpoint saved -> {ckpt_path}")

    return {'losses': losses, 'lr_history': lr_history,
            'contrastive_accs': contrastive_accs}


# ── Embedding Extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, file_list, synapse_map, input_size, device,
                       em_only=True):
    """Get encoder embeddings for every sample (no augmentation)."""
    import cv2
    from torchvision.transforms import functional as TF

    model.eval()
    embeddings, labels = [], []

    for filename in file_list:
        data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
            DATA_DIR,
            DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None,
            filename,
        )

        if em_only:
            # EM-only: single channel, no masks
            mid_z = data_3d.shape[2] // 2
            data_2d = data_3d[:, :, mid_z]
            dmin, dmax = data_2d.min(), data_2d.max()
            data_uint8 = ((data_2d - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
            resized = cv2.resize(data_uint8, (input_size, input_size),
                                 interpolation=cv2.INTER_AREA)
            tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0) / 255.0
            tensor = TF.normalize(tensor, mean=[0.485], std=[0.229])
        else:
            tensor = preprocess_synapse_2d(data_3d, pre_mask_3d, post_mask_3d,
                                           input_size=input_size, augment=False)

        tensor = tensor.unsqueeze(0).to(device)
        h = model.encode(tensor)
        embeddings.append(h.cpu().numpy().flatten())

        syn_id = int(filename.split('_')[0])
        labels.append(1 if synapse_map[syn_id] == 'I' else 0)

    return np.array(embeddings), np.array(labels)


# ── Evaluation Pipeline ──────────────────────────────────────────────────────

def evaluate_representations(embeddings, labels, logger, output_dir=None):
    """UMAP/t-SNE, k-means, linear probe, nearest-neighbour analysis."""
    if output_dir is None:
        output_dir = CONTRASTIVE_FIG_DIR
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    # ── K-means clustering ──
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(emb_scaled)

    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    # Cluster purity
    purity = 0
    for c in range(2):
        mask = cluster_labels == c
        if mask.sum() > 0:
            counts = np.bincount(labels[mask], minlength=2)
            purity += counts.max()
    purity /= len(labels)

    logger.info(f"K-means  ARI: {ari:.4f}  NMI: {nmi:.4f}  Purity: {purity:.4f}")

    # ── Linear probe (5-fold CV) ──
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lp_scores = cross_val_score(lr, emb_scaled, labels, cv=5, scoring='accuracy')
    logger.info(f"Linear probe accuracy: {lp_scores.mean():.4f} (+/- {lp_scores.std():.4f})")

    # ── 5-NN (5-fold CV) ──
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, emb_scaled, labels, cv=5, scoring='accuracy')
    logger.info(f"5-NN accuracy:         {knn_scores.mean():.4f} (+/- {knn_scores.std():.4f})")

    # ── Dimensionality reduction for visualisation ──
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
        emb_2d = reducer.fit_transform(emb_scaled)
        dim_method = 'UMAP'
    except ImportError:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(labels) // 4))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
        emb_2d = reducer.fit_transform(emb_scaled)
        dim_method = 't-SNE'

    # ── Summary figure (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Contrastive Learning — Representation Evaluation',
                 fontsize=16, fontweight='bold')

    colors_ei = ['#2196F3', '#F44336']

    # Panel 1: scatter by E/I label
    ax = axes[0, 0]
    for c, name in [(0, 'E (Excitatory)'), (1, 'I (Inhibitory)')]:
        m = labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors_ei[c],
                   label=name, alpha=0.6, s=20, edgecolors='none')
    ax.set_title(f'{dim_method} — Colored by E/I Label')
    ax.legend(markerscale=2)
    ax.set_xlabel(f'{dim_method}-1')
    ax.set_ylabel(f'{dim_method}-2')

    # Panel 2: scatter by k-means cluster
    ax = axes[0, 1]
    for c in range(2):
        m = cluster_labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors_ei[c],
                   label=f'Cluster {c}', alpha=0.6, s=20, edgecolors='none')
    ax.set_title(f'{dim_method} — Colored by K-means Cluster')
    ax.legend(markerscale=2)
    ax.set_xlabel(f'{dim_method}-1')
    ax.set_ylabel(f'{dim_method}-2')

    # Panel 3: metrics bar chart
    ax = axes[1, 0]
    metric_names = ['ARI', 'NMI', 'Purity', 'Linear\nProbe', '5-NN', 'Chance']
    metric_vals = [ari, nmi, purity, lp_scores.mean(), knn_scores.mean(), 0.5]
    bar_colors = ['#4CAF50'] * 3 + ['#FF9800'] * 2 + ['#9E9E9E']
    bars = ax.bar(metric_names, metric_vals, color=bar_colors)
    ax.set_ylim(0, 1.1)
    ax.set_title('Clustering & Classification Metrics')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    # Panel 4: text summary
    ax = axes[1, 1]
    ax.axis('off')
    if ari > 0.3 and lp_scores.mean() > 0.7:
        interp = ("Strong signal: E/I are clearly distinguishable.\n"
                   "Labels appear reliable — supervised models\n"
                   "need better architecture or training strategy.")
    elif lp_scores.mean() > 0.6:
        interp = ("Moderate signal: some E/I separation exists.\n"
                   "Representations help, but the signal is weak.\n"
                   "Consider stronger augmentation or more data.")
    else:
        interp = ("Weak/no signal: E/I barely separable.\n"
                   "Labels may be noisy, OR the visual\n"
                   "differences are too subtle in middle-Z slices.")

    summary = (
        f"CONTRASTIVE LEARNING RESULTS\n"
        f"{'=' * 42}\n\n"
        f"Clustering (unsupervised):\n"
        f"  Adjusted Rand Index : {ari:+.4f}\n"
        f"  Normalised MI       : {nmi:.4f}\n"
        f"  Cluster Purity      : {purity:.4f}\n\n"
        f"Classification (with labels):\n"
        f"  Linear Probe (5-CV) : {lp_scores.mean():.4f} +/- {lp_scores.std():.4f}\n"
        f"  5-NN         (5-CV) : {knn_scores.mean():.4f} +/- {knn_scores.std():.4f}\n"
        f"  Chance level        : 0.5000\n\n"
        f"Interpretation:\n  {interp}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'contrastive_evaluation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved evaluation figure -> {fig_path}")

    return {
        'ari': ari, 'nmi': nmi, 'purity': purity,
        'linear_probe_mean': lp_scores.mean(),
        'linear_probe_std': lp_scores.std(),
        'knn_mean': knn_scores.mean(),
        'knn_std': knn_scores.std(),
    }


# ── Training / Test Visualization ─────────────────────────────────────────────

def plot_augmented_pairs(dataset, output_dir, n_pairs=6):
    """Show n_pairs synapses with their two augmented views side-by-side."""
    os.makedirs(output_dir, exist_ok=True)

    v1_sample, _, _ = dataset[0]
    n_ch = v1_sample.shape[0]  # 1 (EM-only) or 3

    n_cols = n_ch * 2  # view1 channels + view2 channels
    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(3 * n_cols, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 2:
        # Ensure 2D axes array
        pass

    channel_names = ['EM', 'Pre mask', 'Post mask'][:n_ch]
    for row in range(n_pairs):
        v1, v2, label = dataset[row]
        label_str = 'I' if label == 1 else 'E'
        for ch in range(n_ch):
            # View 1
            ax = axes[row, ch]
            ax.imshow(v1[ch].numpy(), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'View 1: {channel_names[ch]}', fontsize=10)
            if ch == 0:
                ax.set_ylabel(f'#{row} ({label_str})', fontsize=10)

            # View 2
            ax = axes[row, ch + n_ch]
            ax.imshow(v2[ch].numpy(), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'View 2: {channel_names[ch]}', fontsize=10)

    fig.suptitle('Contrastive augmented view pairs\n'
                 '(same synapse, different random augmentations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'augmented_pairs.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_matrix(snapshot, epoch, output_dir):
    """Visualize the 2B x 2B cosine similarity matrix from one batch.

    The matrix has structure:
        [  A(B,B)   P(B,B) ]    A = anchor-anchor similarities
        [  P(B,B)   A(B,B) ]    P = positive-pair block (off-diagonal)
    Positive pairs are at positions (i, i+B) and (i+B, i).
    """
    os.makedirs(output_dir, exist_ok=True)
    sim = snapshot['sim_matrix']
    labels = snapshot['labels']
    B = len(labels)
    N = 2 * B

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: raw similarity matrix (temperature-scaled)
    ax = axes[0]
    im = ax.imshow(sim, cmap='RdBu_r', aspect='equal', vmin=-8, vmax=8)
    ax.set_title(f'Cosine sim / tau  (epoch {epoch})', fontsize=11)
    ax.set_xlabel('Sample index (0..B-1 = view1, B..2B-1 = view2)')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: highlight positive pairs
    ax = axes[1]
    # Build mask: positive pairs are (i, i+B) and (i+B, i)
    pos_mask = np.zeros_like(sim)
    for i in range(B):
        pos_mask[i, i + B] = 1
        pos_mask[i + B, i] = 1
    ax.imshow(sim, cmap='RdBu_r', aspect='equal', vmin=-8, vmax=8)
    # Overlay positive pair locations
    for i in range(B):
        ax.plot(i + B, i, 's', color='lime', markersize=max(1, 80 // B), alpha=0.7)
        ax.plot(i, i + B, 's', color='lime', markersize=max(1, 80 // B), alpha=0.7)
    ax.set_title(f'Positive pairs highlighted (green)', fontsize=11)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')

    # Panel 3: softmax probabilities (what the model "sees" for cross-entropy)
    # Mask diagonal
    diag_mask = np.eye(N, dtype=bool)
    sim_masked = sim.copy()
    sim_masked[diag_mask] = -1e9
    # Softmax per row
    sim_exp = np.exp(sim_masked - sim_masked.max(axis=1, keepdims=True))
    probs = sim_exp / sim_exp.sum(axis=1, keepdims=True)
    ax = axes[2]
    im = ax.imshow(probs, cmap='hot', aspect='equal', vmin=0)
    ax.set_title(f'Softmax probabilities (per row)', fontsize=11)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Mark positive pair targets
    for i in range(B):
        ax.plot(i + B, i, 's', color='cyan', markersize=max(1, 80 // B), alpha=0.7)
        ax.plot(i, i + B, 's', color='cyan', markersize=max(1, 80 // B), alpha=0.7)

    fig.suptitle(f'NT-Xent similarity matrix (batch size B={B}, matrix {N}x{N})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'similarity_matrix_epoch{epoch}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_evolution(snapshots, output_dir):
    """Show how encoder embeddings evolve across training epochs.

    For each snapshot epoch: t-SNE of encoder embeddings colored by E/I label.
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)
    epoch_nums = sorted(snapshots.keys())
    n = len(epoch_nums)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ['#2196F3', '#F44336']
    names = ['E (Excitatory)', 'I (Inhibitory)']

    for ax, ep in zip(axes, epoch_nums):
        snap = snapshots[ep]
        # Combine encoder embeddings from both views
        h = np.concatenate([snap['h1'], snap['h2']], axis=0)
        labs = np.concatenate([snap['labels'], snap['labels']], axis=0)

        scaler = StandardScaler()
        h_scaled = scaler.fit_transform(h)
        perp = min(30, max(5, len(labs) // 4))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        h_2d = tsne.fit_transform(h_scaled)

        for c, name in enumerate(names):
            m = labs == c
            ax.scatter(h_2d[m, 0], h_2d[m, 1], c=colors[c],
                       label=name, alpha=0.6, s=20, edgecolors='none')
        ax.set_title(f'Epoch {ep}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, markerscale=1.5)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    fig.suptitle('Encoder embedding evolution during training\n'
                 '(t-SNE of batch encoder outputs, colored by E/I)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'embedding_evolution.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_contrastive_accuracy(contrastive_accs, output_dir):
    """Plot contrastive identification accuracy over epochs."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(contrastive_accs) + 1)
    ax.plot(epochs, [a * 100 for a in contrastive_accs], 'r-', linewidth=2)
    chance = 100.0 / 64  # ~1.56% for batch_size=64 (1 correct out of 2B-1)
    ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.5, label=f'Chance ({chance:.1f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Contrastive pair identification accuracy\n'
                 '(can the model find its positive pair?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'contrastive_accuracy.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_lr_curve(lr_history, output_dir):
    """Plot learning rate over epochs."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(lr_history) + 1), lr_history, 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ── Static Diagrams ──────────────────────────────────────────────────────────

def plot_architecture_diagram(output_dir):
    """Draw a clear SimCLR architecture diagram (data -> augment -> encode -> project -> loss)."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch
    box_kw = dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1.5)
    box_kw_small = dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.2)
    box_kw_loss = dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', linewidth=1.5)

    # Boxes first (so text draws on top)
    ax.add_patch(FancyBboxPatch((0.5, 3.5), 2, 1.2, **box_kw))
    ax.add_patch(FancyBboxPatch((3.2, 3.5), 1.6, 1.2, **box_kw_small))
    ax.add_patch(FancyBboxPatch((5.2, 3.7), 1.2, 0.5, **box_kw_small))
    ax.add_patch(FancyBboxPatch((5.2, 2.9), 1.2, 0.5, **box_kw_small))
    ax.add_patch(FancyBboxPatch((7.2, 2.8), 2, 1.6, **box_kw))
    ax.add_patch(FancyBboxPatch((9.7, 2.8), 1.6, 1.6, **box_kw))
    ax.add_patch(FancyBboxPatch((11.8, 3.7), 0.8, 0.4, **box_kw_small))
    ax.add_patch(FancyBboxPatch((11.8, 2.9), 0.8, 0.4, **box_kw_small))
    ax.add_patch(FancyBboxPatch((12.8, 3.0), 1.4, 1.2, **box_kw_loss))

    # Arrows
    for (x1, x2, y) in [(2.5, 3.5, 4.1), (4.8, 5.2, 4.1), (6.4, 6.8, 4.0), (6.4, 6.8, 3.2),
                         (6.8, 7.2, 3.6), (9.2, 9.7, 3.6), (11.3, 11.8, 3.6), (12.6, 13.0, 3.6)]:
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Labels
    ax.text(1.5, 4.1, 'Synapse\n(3ch: EM + pre + post)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4, 4.1, 'Augment\n(x2)', ha='center', va='center', fontsize=9)
    ax.text(5.8, 4.0, 'View 1', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5.8, 3.2, 'View 2', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8.2, 3.6, 'Encoder (CNN)\nshared weights', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10.5, 3.6, 'Projector (MLP)\nL2-norm', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(12.2, 4.0, 'z1', ha='center', va='center', fontsize=9)
    ax.text(12.2, 3.2, 'z2', ha='center', va='center', fontsize=9)
    ax.text(13.5, 3.6, 'NT-Xent\nLoss', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_title('SimCLR contrastive learning architecture', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'architecture_diagram.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_test_process(output_dir):
    """One figure summarizing training and evaluation process."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    from matplotlib.patches import FancyBboxPatch
    box = dict(boxstyle='round,pad=0.25', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.2)

    # Left: Training
    ax = axes[0]
    steps = [
        (5, 8.5, '1. Load synapse (3ch)'),
        (5, 7.2, '2. Two random augmentations'),
        (5, 5.9, '3. Forward: Encoder -> Projector -> z1, z2'),
        (5, 4.6, '4. NT-Xent loss (pull pairs, push others)'),
        (5, 3.3, '5. Backward + optimizer step'),
    ]
    for x, y, t in steps:
        ax.add_patch(FancyBboxPatch((x - 2.2, y - 0.35), 4.4, 0.5, **box))
        ax.text(x, y, t, ha='center', va='center', fontsize=10)
    ax.text(5, 9.5, 'Training (per batch)', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 1.5, 'Repeat over epochs;\nReduceLROnPlateau.', ha='center', fontsize=9, style='italic')

    # Right: Test / evaluation
    ax = axes[1]
    steps = [
        (5, 8.5, '1. Load synapse (no augment)'),
        (5, 7.2, '2. Forward: Encoder only (no projector)'),
        (5, 5.9, '3. Collect embeddings for all samples'),
        (5, 4.6, '4. K-means: ARI, NMI, Purity'),
        (5, 3.3, '5. Linear probe & 5-NN (5-fold CV)'),
    ]
    for x, y, t in steps:
        ax.add_patch(FancyBboxPatch((x - 2.2, y - 0.35), 4.4, 0.5, **box))
        ax.text(x, y, t, ha='center', va='center', fontsize=10)
    ax.text(5, 9.5, 'Evaluation (after training)', ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 1.5, 'UMAP/t-SNE + metrics\nsaved in evaluation figure.', ha='center', fontsize=9, style='italic')

    fig.suptitle('Contrastive pipeline: training and test process', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'training_test_process.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_contrastive_loss(losses, output_dir):
    """Save a contrastive-loss curve plot."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NT-Xent Loss')
    ax.set_title('Contrastive Pretraining Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'contrastive_loss.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SimCLR Contrastive Learning for Synapses')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--cnn_depth', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for quick test (default: use all)')
    parser.add_argument('--resume', nargs='?', const='latest', default=None,
                        help='Resume from checkpoint (path or "latest")')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help='Save checkpoint every N epochs (0=disabled)')
    parser.add_argument('--em_only', action='store_true', default=True,
                        help='Use EM channel only (default: True)')
    parser.add_argument('--multichannel', action='store_true', default=False,
                        help='Use all 3 channels (EM + pre + post masks)')
    parser.add_argument('--multi_slice', action='store_true', default=False,
                        help='Use different Z-slices from same synapse as view pair (instead of same slice + augmentation)')

    args = parser.parse_args()
    em_only = not args.multichannel  # --multichannel overrides default em_only
    in_channels = 1 if em_only else 3

    model_name = 'contrastive'
    logger = setup_logging(model_name)
    logger.info(f"Starting SimCLR Contrastive Learning ({'EM-only' if em_only else '3-channel'}, "
                f"{'multi_slice' if args.multi_slice else 'single-slice+augment'})")

    set_random_seeds(42)
    device = get_device(prefer_gpu=not args.cpu)

    # ── Data — use ALL samples (unsupervised, no train/test split needed) ──
    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(
        logger=logger)
    all_files = train_files + test_files
    if args.max_samples is not None:
        all_files = all_files[: args.max_samples]
        logger.info(f"Using subset: {len(all_files)} samples (--max_samples)")
    logger.info(f"Total samples for contrastive pretraining: {len(all_files)}")

    # Dataset & dataloader — prefer extracted files on disk, skip archive
    data_on_disk = os.path.isdir(DATA_DIR) and any(
        f.endswith('.npy') for f in os.listdir(DATA_DIR)[:5])
    logger.info(f"Data source: {'disk' if data_on_disk else '7z archive'}")

    dataset = ContrastiveSynapseDataset(
        all_files, synapse_map, input_size=args.input_size,
        archive_path=None if data_on_disk else DATA_ARCHIVE,
        em_only=em_only, use_multi_slice=args.multi_slice)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ── Model ──
    model = SimCLRModel(cnn_depth=args.cnn_depth,
                        proj_dim=args.proj_dim,
                        in_channels=in_channels).to(device)
    print_model_summary(model, logger=logger)

    # ── Optimiser, scheduler, loss ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = NTXentLoss(temperature=args.temperature)

    logger.info(f"Config: epochs={args.epochs}  bs={args.batch_size}  "
                f"temp={args.temperature}  lr={args.lr}")

    # ── Resume from checkpoint ──
    start_epoch = 0
    prev_losses = []
    prev_lr_history = []

    if args.resume:
        ckpt_path = (get_latest_checkpoint() if args.resume == 'latest'
                     else args.resume)
        if ckpt_path and os.path.isfile(ckpt_path):
            start_epoch, prev_losses, prev_lr_history = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, device, logger)
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at '{ckpt_path}', starting fresh")

    for d in [CONTRASTIVE_FIG_DIR, CONTRASTIVE_DIAGRAMS_DIR,
              CONTRASTIVE_TRAINING_DIR, CONTRASTIVE_EVAL_DIR, CONTRASTIVE_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    plot_architecture_diagram(CONTRASTIVE_DIAGRAMS_DIR)
    plot_training_test_process(CONTRASTIVE_DIAGRAMS_DIR)

    # Visualize augmented pairs before training
    plot_augmented_pairs(dataset, CONTRASTIVE_DATA_DIR, n_pairs=6)
    logger.info("Saved data/augmented_pairs.png")

    # ── Per-epoch eval function (closure over data) ──
    # Use a subsample for fast per-epoch eval (full eval at the end)
    eval_sample_size = min(1000, len(all_files))
    eval_files = all_files[:eval_sample_size]
    all_snapshots = {}  # accumulate for embedding evolution plot

    def epoch_eval_fn(mdl, epoch_num, loss_hist, lr_hist, snapshot,
                      contrastive_accs):
        """Evaluate model and update all plots at end of each epoch."""
        # Training curves
        plot_contrastive_loss(loss_hist, CONTRASTIVE_TRAINING_DIR)
        plot_lr_curve(lr_hist, CONTRASTIVE_TRAINING_DIR)
        plot_contrastive_accuracy(contrastive_accs, CONTRASTIVE_TRAINING_DIR)

        # Similarity matrix for this epoch (overwrite latest)
        if snapshot is not None:
            plot_similarity_matrix(snapshot, epoch_num, CONTRASTIVE_TRAINING_DIR)
            # Keep snapshots for first, every 5th, and last for evolution plot
            if epoch_num == 1 or epoch_num % 5 == 0 or epoch_num == args.epochs:
                all_snapshots[epoch_num] = snapshot
            if len(all_snapshots) >= 2:
                plot_embedding_evolution(all_snapshots, CONTRASTIVE_TRAINING_DIR)

        # Classification metrics
        embs, labs = extract_embeddings(
            mdl, eval_files, synapse_map, args.input_size, device,
            em_only=em_only)
        results = evaluate_representations(embs, labs, logger, CONTRASTIVE_EVAL_DIR)
        lp = results['linear_probe_mean']
        knn = results['knn_mean']
        logger.info(f"  >> Epoch {epoch_num} | Linear probe {lp:.1%} | "
                     f"5-NN {knn:.1%}")

    # ── Train ──
    train_results = train_contrastive(
        model, dataloader, optimizer, scheduler, criterion,
        args.epochs, device, logger,
        start_epoch=start_epoch, prev_losses=prev_losses,
        prev_lr_history=prev_lr_history,
        checkpoint_every=args.checkpoint_every,
        eval_fn=epoch_eval_fn)

    losses = train_results['losses']
    lr_history = train_results['lr_history']
    contrastive_accs = train_results['contrastive_accs']

    # Final training plots
    plot_contrastive_loss(losses, CONTRASTIVE_TRAINING_DIR)
    plot_lr_curve(lr_history, CONTRASTIVE_TRAINING_DIR)
    plot_contrastive_accuracy(contrastive_accs, CONTRASTIVE_TRAINING_DIR)
    if all_snapshots:
        plot_embedding_evolution(all_snapshots, CONTRASTIVE_TRAINING_DIR)

    # ── Save encoder ──
    os.makedirs('saved_models', exist_ok=True)
    save_path = 'saved_models/contrastive_encoder.pth'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model -> {save_path}")

    # ── Full evaluation on all data ──
    logger.info("Extracting embeddings for full evaluation ...")
    embeddings, labels = extract_embeddings(
        model, all_files, synapse_map, args.input_size, device,
        em_only=em_only)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    results = evaluate_representations(embeddings, labels, logger, CONTRASTIVE_EVAL_DIR)
    logger.info("Done - contrastive learning pipeline complete.")


if __name__ == '__main__':
    main()
