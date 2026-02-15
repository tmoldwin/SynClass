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


# ── Contrastive Augmentation ─────────────────────────────────────────────────

def contrastive_augment(data_3d, pre_mask_3d, post_mask_3d, input_size=224, seed=None):
    """Strong augmentation for contrastive learning. Returns one augmented view.

    Geometric transforms apply to all 3 channels.
    Intensity transforms (jitter, blur, noise) apply only to the EM channel.
    """
    import cv2
    from torchvision.transforms import functional as TF

    rng = np.random.RandomState(seed)

    mid_z = data_3d.shape[2] // 2
    data_2d = data_3d[:, :, mid_z].copy()
    pre_2d = pre_mask_3d[:, :, mid_z].copy()
    post_2d = post_mask_3d[:, :, mid_z].copy()

    # Normalize EM to [0, 255]
    dmin, dmax = data_2d.min(), data_2d.max()
    data_uint8 = ((data_2d - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
    pre_uint8 = (pre_2d * 255).astype(np.uint8) if pre_2d.max() <= 1.0 else pre_2d.astype(np.uint8)
    post_uint8 = (post_2d * 255).astype(np.uint8) if post_2d.max() <= 1.0 else post_2d.astype(np.uint8)

    # 1. Random flips
    if rng.random() > 0.5:
        data_uint8 = np.flipud(data_uint8).copy()
        pre_uint8 = np.flipud(pre_uint8).copy()
        post_uint8 = np.flipud(post_uint8).copy()
    if rng.random() > 0.5:
        data_uint8 = np.fliplr(data_uint8).copy()
        pre_uint8 = np.fliplr(pre_uint8).copy()
        post_uint8 = np.fliplr(post_uint8).copy()

    # 2. Random resized crop with rotation (scale 0.5–1.0, stronger than supervised)
    h, w = data_uint8.shape
    crop_ratio = rng.uniform(0.5, 1.0)
    crop_size = int(min(h, w) * crop_ratio)
    angle = rng.uniform(-45, 45)
    margin = crop_size // 2
    center_h = rng.randint(margin, max(margin + 1, h - margin))
    center_w = rng.randint(margin, max(margin + 1, w - margin))

    data_uint8 = rotated_crop(data_uint8, center_h, center_w, crop_size, angle)
    pre_uint8 = rotated_crop(pre_uint8, center_h, center_w, crop_size, angle)
    post_uint8 = rotated_crop(post_uint8, center_h, center_w, crop_size, angle)

    # 3. Intensity jitter on EM only (brightness + contrast)
    brightness = rng.uniform(0.7, 1.3)
    contrast = rng.uniform(0.7, 1.3)
    data_f = data_uint8.astype(np.float32)
    mean_val = data_f.mean()
    data_f = (data_f - mean_val) * contrast + mean_val * brightness
    data_uint8 = np.clip(data_f, 0, 255).astype(np.uint8)

    # 4. Gaussian blur on EM only (50 % chance)
    if rng.random() > 0.5:
        ksize = int(rng.choice([3, 5]))
        data_uint8 = cv2.GaussianBlur(data_uint8, (ksize, ksize), 0)

    # 5. Gaussian noise on EM only
    noise = rng.normal(0, 0.05 * 255, data_uint8.shape).astype(np.int16)
    data_uint8 = np.clip(data_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Resize
    data_resized = cv2.resize(data_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)
    pre_resized = cv2.resize(pre_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    post_resized = cv2.resize(post_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

    # To tensor + normalise (same as preprocessing.py)
    data_t = torch.tensor(data_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    pre_t = torch.tensor(pre_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    post_t = torch.tensor(post_resized, dtype=torch.float32).unsqueeze(0) / 255.0
    data_t = TF.normalize(data_t, mean=[0.485], std=[0.229])

    return torch.cat([data_t, pre_t, post_t], dim=0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class ContrastiveSynapseDataset(Dataset):
    """Returns two differently-augmented views of each synapse (+ label for eval)."""

    def __init__(self, file_list, synapse_map, input_size=224,
                 data_dir=None, archive_path=None):
        from constants import DATA_DIR, DATA_ARCHIVE
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.input_size = input_size
        self.data_dir = data_dir or DATA_DIR
        self.archive_path = (
            archive_path if archive_path and os.path.isfile(archive_path)
            else (DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None)
        )
        print(f"ContrastiveDataset: {len(file_list)} synapses")

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

        # Truly random seeds per call (different every epoch)
        seed1 = int.from_bytes(os.urandom(4), 'big')
        seed2 = int.from_bytes(os.urandom(4), 'big')

        view1 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                    self.input_size, seed=seed1)
        view2 = contrastive_augment(data_3d, pre_mask_3d, post_mask_3d,
                                    self.input_size, seed=seed2)

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

    def __init__(self, cnn_depth=3, proj_dim=64, hidden_dim=256):
        super().__init__()
        from synapse_classifier_2dcnn_multichannel import CNN2DMultiChannel
        backbone = CNN2DMultiChannel(num_classes=2, dropout_rate=0.0,
                                     cnn_depth=cnn_depth)
        self.encoder = backbone.features

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

def train_contrastive(model, dataloader, optimizer, scheduler, criterion,
                      epochs, device, logger):
    """Train encoder with NT-Xent. Returns loss history."""
    model.train()
    losses = []
    total_batches = len(dataloader)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        log_interval = max(1, total_batches // 5)  # ~5 prints per epoch

        for view1, view2, _ in dataloader:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % log_interval == 0 or n_batches == total_batches:
                running_avg = epoch_loss / n_batches
                logger.info(f"  [{n_batches:4d}/{total_batches}] loss: {loss.item():.4f}  avg: {running_avg:.4f}")
                sys.stdout.flush()

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if scheduler is not None:
            scheduler.step(avg_loss)

        lr_now = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1:3d}/{epochs} | "
                     f"Loss {avg_loss:.4f} | LR {lr_now:.2e}")

    return losses


# ── Embedding Extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, file_list, synapse_map, input_size, device):
    """Get encoder embeddings for every sample (no augmentation)."""
    model.eval()
    embeddings, labels = [], []

    for filename in file_list:
        data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
            DATA_DIR,
            DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None,
            filename,
        )
        tensor = preprocess_synapse_2d(data_3d, pre_mask_3d, post_mask_3d,
                                       input_size=input_size, augment=False)
        tensor = tensor.unsqueeze(0).to(device)

        h = model.encode(tensor)
        embeddings.append(h.cpu().numpy().flatten())

        syn_id = int(filename.split('_')[0])
        labels.append(1 if synapse_map[syn_id] == 'I' else 0)

    return np.array(embeddings), np.array(labels)


# ── Evaluation Pipeline ──────────────────────────────────────────────────────

def evaluate_representations(embeddings, labels, logger, output_dir='figures'):
    """UMAP/t-SNE, k-means, linear probe, nearest-neighbour analysis."""
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
    logger.info(f"Saved evaluation figure → {fig_path}")

    return {
        'ari': ari, 'nmi': nmi, 'purity': purity,
        'linear_probe_mean': lp_scores.mean(),
        'linear_probe_std': lp_scores.std(),
        'knn_mean': knn_scores.mean(),
        'knn_std': knn_scores.std(),
    }


def plot_contrastive_loss(losses, output_dir='figures'):
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

    args = parser.parse_args()

    model_name = 'contrastive'
    logger = setup_logging(model_name)
    logger.info("Starting SimCLR Contrastive Learning")

    set_random_seeds(42)
    device = get_device(prefer_gpu=not args.cpu)

    # ── Data — use ALL samples (unsupervised, no train/test split needed) ──
    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(
        logger=logger)
    all_files = train_files + test_files
    logger.info(f"Total samples for contrastive pretraining: {len(all_files)}")

    # Dataset & dataloader — prefer extracted files on disk, skip archive
    data_on_disk = os.path.isdir(DATA_DIR) and any(
        f.endswith('.npy') for f in os.listdir(DATA_DIR)[:5])
    logger.info(f"Data source: {'disk' if data_on_disk else '7z archive'}")

    dataset = ContrastiveSynapseDataset(
        all_files, synapse_map, input_size=args.input_size,
        archive_path=None if data_on_disk else DATA_ARCHIVE)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ── Model ──
    model = SimCLRModel(cnn_depth=args.cnn_depth,
                        proj_dim=args.proj_dim).to(device)
    print_model_summary(model, logger=logger)

    # ── Optimiser, scheduler, loss ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = NTXentLoss(temperature=args.temperature)

    logger.info(f"Config: epochs={args.epochs}  bs={args.batch_size}  "
                f"temp={args.temperature}  lr={args.lr}")

    # ── Train ──
    losses = train_contrastive(model, dataloader, optimizer, scheduler,
                               criterion, args.epochs, device, logger)
    plot_contrastive_loss(losses)

    # ── Save encoder ──
    os.makedirs('saved_models', exist_ok=True)
    save_path = 'saved_models/contrastive_encoder.pth'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model → {save_path}")

    # ── Evaluate representations ──
    logger.info("Extracting embeddings for evaluation …")
    embeddings, labels = extract_embeddings(
        model, all_files, synapse_map, args.input_size, device)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    results = evaluate_representations(embeddings, labels, logger)
    logger.info("Done — contrastive learning pipeline complete.")


if __name__ == '__main__':
    main()
