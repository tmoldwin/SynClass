"""
Convolutional autoencoder for synapse representation learning.
Trains unsupervised with reconstruction loss; evaluates E/I separability via linear probe and k-NN.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import setup_logging, DATA_DIR, DATA_ARCHIVE
from data_loader import prepare_synapse_data
from preprocessing import create_multichannel_dataloaders, preprocess_synapse_2d, load_synapse_data
from utils import set_random_seeds, get_device, print_model_summary, save_checkpoint, load_checkpoint

FIG_DIR = os.path.join('figures', 'autoencoder')
FIG_DIAGRAMS = os.path.join(FIG_DIR, 'diagrams')
FIG_TRAINING = os.path.join(FIG_DIR, 'training')
FIG_EVAL = os.path.join(FIG_DIR, 'evaluation')
FIG_DATA = os.path.join(FIG_DIR, 'data')
CHECKPOINT_DIR = 'checkpoints_autoencoder'


# ── Model ───────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder. Encoder mirrors CNN2DMultiChannel style; decoder uses ConvTranspose."""

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # Encoder: 224 -> 112 -> 56 -> 28 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
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
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.enc_fc = nn.Linear(256 * 4 * 4, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_final = nn.Conv2d(32, in_channels, kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        h = self.decoder(h)
        h = F.interpolate(h, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.dec_final(h)
        return out

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ── Data: EM-only slice from multichannel dataset ─────────────────────────────

def get_em_only_batch(batch):
    """Return (x_em, label) where x_em is first channel only, in [0,1] for reconstruction."""
    x, y = batch
    x_em = x[:, :1].clone()
    return x_em, y


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, em_only, logger=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        if em_only:
            x, _ = get_em_only_batch(batch)
        else:
            x = batch[0]
        x = x.to(device)
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_autoencoder(model, train_loader, device, epochs, lr, latent_dim,
                      em_only, checkpoint_every, save_dir, logger,
                      train_files=None, test_files=None, synapse_map=None, input_size=224,
                      eval_fn=None, prev_epoch=0, prev_losses=None, prev_lr_history=None,
                      prev_accuracy_history=None, accuracy_every=1, outputs_csv_path=None,
                      plot_loss_acc_fn=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = list(prev_losses) if prev_losses else []
    lr_history = list(prev_lr_history) if prev_lr_history else []
    accuracy_history = list(prev_accuracy_history) if prev_accuracy_history else []

    def save_ckpt(epoch, loss_val, acc_val):
        meta = {
            'scheduler': scheduler.state_dict(), 'losses': losses, 'lr_history': lr_history,
            'accuracy_history': accuracy_history,
        }
        path = os.path.join(save_dir, f"ae_epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, loss_val, path, metadata=meta)
        if outputs_csv_path:
            with open(outputs_csv_path, 'w') as f:
                f.write("epoch,loss,linear_probe_accuracy\n")
                for i, (l, a) in enumerate(zip(losses, accuracy_history)):
                    f.write(f"{i+1},{l:.6f},{a:.4f}\n")
        if plot_loss_acc_fn and losses and accuracy_history:
            plot_loss_acc_fn(losses, accuracy_history)

    for epoch in range(prev_epoch, epochs):
        t0 = time.perf_counter()
        loss = train_epoch(model, train_loader, criterion, optimizer, device, em_only, logger)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        losses.append(loss)
        lr_history.append(current_lr)

        acc = None
        if train_files and test_files and synapse_map and (epoch + 1) % accuracy_every == 0:
            embs_train, labs_train = extract_embeddings(model, train_files, synapse_map, device, input_size, em_only)
            embs_test, labs_test = extract_embeddings(model, test_files, synapse_map, device, input_size, em_only)
            acc = compute_linear_probe_accuracy(embs_train, labs_train, embs_test, labs_test)
            accuracy_history.append(acc)
        else:
            accuracy_history.append(accuracy_history[-1] if accuracy_history else 0.0)
        acc = accuracy_history[-1]

        elapsed = time.perf_counter() - t0
        if logger:
            logger.info(f"Epoch {epoch+1}/{epochs}  loss={loss:.6f}  acc(E/I)={acc:.4f}  lr={current_lr:.2e}  time={elapsed:.1f}s")
        else:
            print(f"Epoch {epoch+1}/{epochs}  loss={loss:.6f}  acc(E/I)={acc:.4f}  lr={current_lr:.2e}  time={elapsed:.1f}s")

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_ckpt(epoch + 1, loss, acc)
            if logger:
                logger.info(f"Checkpoint saved -> {save_dir}/ae_epoch_{epoch+1}.pth")

        if outputs_csv_path:
            with open(outputs_csv_path, 'w') as f:
                f.write("epoch,loss,linear_probe_accuracy\n")
                for i, (l, a) in enumerate(zip(losses, accuracy_history)):
                    f.write(f"{i+1},{l:.6f},{a:.4f}\n")

        if eval_fn and (epoch + 1) % max(1, epochs // 5) == 0:
            eval_fn(epoch + 1)

    path_latest = os.path.join(save_dir, 'ae_latest.pth')
    save_checkpoint(model, optimizer, epochs, losses[-1] if losses else 0.0, path_latest,
                   metadata={'scheduler': scheduler.state_dict(), 'losses': losses, 'lr_history': lr_history,
                             'accuracy_history': accuracy_history})
    if outputs_csv_path:
        with open(outputs_csv_path, 'w') as f:
            f.write("epoch,loss,linear_probe_accuracy\n")
            for i, (l, a) in enumerate(zip(losses, accuracy_history)):
                f.write(f"{i+1},{l:.6f},{a:.4f}\n")
    if logger:
        logger.info(f"Latest checkpoint saved -> {path_latest}")
    return losses, lr_history, accuracy_history


def get_latest_checkpoint(save_dir):
    """Return path to latest ae checkpoint in save_dir, or None."""
    if not os.path.isdir(save_dir):
        return None
    candidates = [f for f in os.listdir(save_dir) if f.startswith('ae_') and f.endswith('.pth')]
    if not candidates:
        return None
    def epoch_num(f):
        if 'epoch' in f:
            try:
                return int(f.replace('ae_epoch_', '').replace('.pth', ''))
            except ValueError:
                return 0
        return 0
    latest = max(candidates, key=epoch_num)
    return os.path.join(save_dir, latest)


# ── Embedding extraction ─────────────────────────────────────────────────────

def extract_embeddings(model, file_list, synapse_map, device, input_size=224, em_only=True):
    """Extract latent vectors for all files (no labels used in model)."""
    import cv2
    from torchvision.transforms import functional as TF
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for filename in file_list:
            try:
                data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
                    DATA_DIR,
                    DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None,
                    filename,
                )
            except Exception:
                continue
            if em_only:
                mid_z = data_3d.shape[2] // 2
                data_2d = data_3d[:, :, mid_z]
                dmin, dmax = data_2d.min(), data_2d.max()
                data_uint8 = ((data_2d - dmin) / (dmax - dmin + 1e-8) * 255).astype(np.uint8)
                resized = cv2.resize(data_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)
                tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                tensor = TF.normalize(tensor, mean=[0.485], std=[0.229])
            else:
                tensor = preprocess_synapse_2d(data_3d, pre_mask_3d, post_mask_3d,
                                               input_size=input_size, augment=False)
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)
            z = model.encode(tensor)
            embeddings.append(z.cpu().numpy().flatten())
            syn_id = int(filename.split('_')[0])
            labels.append(1 if synapse_map[syn_id] == 'I' else 0)
    return np.array(embeddings), np.array(labels)


def compute_linear_probe_accuracy(emb_train, lab_train, emb_test, lab_test):
    """Fit logistic regression on train embeddings, return accuracy on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    emb_train_s = scaler.fit_transform(emb_train)
    emb_test_s = scaler.transform(emb_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(emb_train_s, lab_train)
    return float(lr.score(emb_test_s, lab_test))


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_representations(embeddings, labels, logger, output_dir=None):
    """Linear probe, 5-NN, k-means ARI/NMI/Purity, t-SNE/UMAP."""
    if output_dir is None:
        output_dir = FIG_EVAL
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(emb_scaled)
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    purity = 0
    for c in range(2):
        mask = cluster_labels == c
        if mask.sum() > 0:
            counts = np.bincount(labels[mask], minlength=2)
            purity += counts.max()
    purity /= len(labels)

    logger.info(f"K-means  ARI: {ari:.4f}  NMI: {nmi:.4f}  Purity: {purity:.4f}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lp_scores = cross_val_score(lr, emb_scaled, labels, cv=5, scoring='accuracy')
    logger.info(f"Linear probe accuracy: {lp_scores.mean():.4f} (+/- {lp_scores.std():.4f})")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, emb_scaled, labels, cv=5, scoring='accuracy')
    logger.info(f"5-NN accuracy:         {knn_scores.mean():.4f} (+/- {knn_scores.std():.4f})")

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

    colors_ei = ['#2196F3', '#F44336']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Autoencoder — Representation Evaluation', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    for c, name in [(0, 'E (Excitatory)'), (1, 'I (Inhibitory)')]:
        m = labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors_ei[c], label=name, alpha=0.6, s=20, edgecolors='none')
    ax.set_title(f'{dim_method} — Colored by E/I Label')
    ax.legend(markerscale=2)

    ax = axes[0, 1]
    for c in range(2):
        m = cluster_labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors_ei[c], label=f'Cluster {c}', alpha=0.6, s=20, edgecolors='none')
    ax.set_title(f'{dim_method} — K-means Cluster')
    ax.legend(markerscale=2)

    ax = axes[1, 0]
    metric_names = ['ARI', 'NMI', 'Purity', 'Linear\nProbe', '5-NN', 'Chance']
    metric_vals = [ari, nmi, purity, lp_scores.mean(), knn_scores.mean(), 0.5]
    bars = ax.bar(metric_names, metric_vals, color=['#4CAF50'] * 3 + ['#FF9800'] * 2 + ['#9E9E9E'])
    ax.set_ylim(0, 1.1)
    ax.set_title('Clustering & Classification Metrics')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=10)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"AUTOENCODER RESULTS\n{'=' * 42}\n\n"
        f"Clustering: ARI={ari:.4f}  NMI={nmi:.4f}  Purity={purity:.4f}\n\n"
        f"Linear Probe (5-CV): {lp_scores.mean():.4f} +/- {lp_scores.std():.4f}\n"
        f"5-NN         (5-CV): {knn_scores.mean():.4f} +/- {knn_scores.std():.4f}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'autoencoder_evaluation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved evaluation figure -> {fig_path}")

    return {
        'ari': ari, 'nmi': nmi, 'purity': purity,
        'linear_probe_mean': lp_scores.mean(), 'linear_probe_std': lp_scores.std(),
        'knn_mean': knn_scores.mean(), 'knn_std': knn_scores.std(),
        'emb_2d': emb_2d, 'dim_method': dim_method,
    }


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_architecture_diagram(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.text(5, 5.5, 'Conv Autoencoder', fontsize=16, fontweight='bold', ha='center')
    ax.text(2, 4, 'Input\n(1 or 3 ch)\n224x224', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.annotate('', xy=(4, 4), xytext=(2.8, 4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 4, 'Encoder\nConv+Pool\n-> 4x4x256', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.annotate('', xy=(7.2, 4), xytext=(5.2, 4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(8, 4, 'Latent\nz (128)', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    ax.annotate('', xy=(7.8, 3), xytext=(8, 3.6), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(8, 2.5, 'Decoder\nLinear+ConvT\n+ Interp', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.annotate('', xy=(5.2, 2.5), xytext=(7.8, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 2.5, 'Recon\n224x224', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.annotate('', xy=(2.2, 4), xytext=(3.8, 4), arrowprops=dict(arrowstyle='->', lw=2))
    plt.tight_layout()
    path = os.path.join(output_dir, 'architecture.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curve(losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(losses, color='#1976D2', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction loss (MSE)')
    ax.set_title('Autoencoder training loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_and_accuracy(losses, accuracy_history, output_dir):
    """Single figure: loss and E/I classification accuracy over epochs."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    epochs = range(1, len(losses) + 1)
    ax1.plot(epochs, losses, color='#1976D2', lw=2)
    ax1.set_ylabel('Reconstruction loss (MSE)')
    ax1.set_title('Training loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, accuracy_history, color='#388E3C', lw=2)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_title('E/I classification (linear probe on latent)')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_and_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_lr_curve(lr_history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(lr_history, color='#388E3C', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    ax.set_title('Learning rate schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstructions(model, loader, device, epoch, output_dir, em_only, n=8):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    in_ch = 1 if em_only else 3
    rows = n
    cols = 2 * in_ch
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if np.isscalar(axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if rows == 1 else axes.reshape(-1, 1)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n:
                break
            if em_only:
                x, _ = get_em_only_batch(batch)
            else:
                x = batch[0]
            x = x.to(device)
            recon, _ = model(x)
            x = x.cpu()
            recon = recon.cpu()
            for ch in range(in_ch):
                ax = axes[i, ch]
                ax.imshow(x[0, ch].numpy(), cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title('Input' if in_ch == 1 else ['EM', 'Pre', 'Post'][ch])
                ax = axes[i, in_ch + ch]
                ax.imshow(recon[0, ch].numpy(), cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title('Recon')
    fig.suptitle(f'Reconstructions (epoch {epoch})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'reconstructions_epoch_{epoch}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(emb_2d, labels, dim_method, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    colors_ei = ['#2196F3', '#F44336']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for c, name in [(0, 'E'), (1, 'I')]:
        m = labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=colors_ei[c], label=name, alpha=0.6, s=20)
    ax.set_title(f'Latent space ({dim_method})')
    ax.legend()
    ax.set_xlabel(f'{dim_method}-1')
    ax.set_ylabel(f'{dim_method}-2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'latent_{dim_method.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_inputs(loader, output_dir, em_only, n=6):
    os.makedirs(output_dir, exist_ok=True)
    in_ch = 1 if em_only else 3
    fig, axes = plt.subplots(n, in_ch, figsize=(2 * in_ch, 2 * n))
    if np.isscalar(axes) or axes.ndim == 1:
        axes = np.atleast_2d(axes) if axes.ndim == 1 else np.array([[axes]])
    if n == 1 and in_ch > 1:
        axes = axes.reshape(1, -1)
    elif in_ch == 1 and n > 1:
        axes = axes.reshape(-1, 1)
    for i, batch in enumerate(loader):
        if i >= n:
            break
        if em_only:
            x, _ = get_em_only_batch(batch)
        else:
            x = batch[0]
        for ch in range(in_ch):
            ax = axes[i, ch]
            ax.imshow(x[0, ch].numpy(), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title('EM' if in_ch == 1 else ['EM', 'Pre', 'Post'][ch])
    fig.suptitle('Sample inputs', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_inputs.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Convolutional autoencoder for synapse representation learning')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--em_only', action='store_true', default=True, help='Use only EM channel')
    parser.add_argument('--multichannel', action='store_true', help='Use 3 channels (overrides em_only)')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_name', type=str, default='autoencoder')
    args = parser.parse_args()

    em_only = not args.multichannel
    in_channels = 1 if em_only else 3

    for d in [FIG_DIR, FIG_DIAGRAMS, FIG_TRAINING, FIG_EVAL, FIG_DATA, CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(args.run_name)
    logger.info("Autoencoder: in_channels=%s, latent_dim=%s", in_channels, args.latent_dim)
    set_random_seeds(42)
    device = get_device(prefer_gpu=not args.cpu)

    train_files, test_files, synapse_map, data_stats = prepare_synapse_data(logger=logger)
    if args.max_samples:
        n = min(args.max_samples, len(train_files))
        train_files = train_files[:n]
        test_files = test_files[: min(args.max_samples, len(test_files))]
        logger.info(f"Using max_samples={n} train and {len(test_files)} test files")
    all_files = train_files + test_files

    train_loader, val_loader = create_multichannel_dataloaders(
        train_files, test_files, synapse_map,
        batch_size=args.batch_size, num_workers=4, pin_memory=True,
        input_size=args.input_size, augment_train=False, examples_per_epoch=None,
        augment_val=False, val_examples_per_epoch=None,
    )

    model = ConvAutoencoder(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    print_model_summary(model, logger=logger)

    plot_architecture_diagram(FIG_DIAGRAMS)
    plot_sample_inputs(train_loader, FIG_DATA, em_only)

    start_epoch = 0
    prev_losses, prev_lr_history = None, None
    prev_accuracy_history = None
    if args.resume:
        ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path and os.path.isfile(ckpt_path):
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
            ckpt = load_checkpoint(ckpt_path, model, optimizer=optimizer, device=device)
            start_epoch = ckpt.get('epoch', 0)
            prev_losses = ckpt.get('losses', [])
            prev_lr_history = ckpt.get('lr_history', [])
            prev_accuracy_history = ckpt.get('accuracy_history', [])
            logger.info(f"Resumed from epoch {start_epoch}")

    def eval_fn(epoch):
        embs_train, labs_train = extract_embeddings(model, train_files, synapse_map, device,
                                                    args.input_size, em_only)
        embs_test, labs_test = extract_embeddings(model, test_files, synapse_map, device,
                                                  args.input_size, em_only)
        embs = np.concatenate([embs_train, embs_test])
        labs = np.concatenate([labs_train, labs_test])
        res = evaluate_representations(embs, labs, logger, FIG_EVAL)
        plot_latent_space(res['emb_2d'], labs, res['dim_method'], FIG_EVAL)
        plot_reconstructions(model, val_loader, device, epoch, FIG_TRAINING, em_only)

    outputs_csv_path = os.path.join(FIG_DIR, 'training_outputs.csv')
    losses = []
    lr_history = []
    accuracy_history = []
    if start_epoch < args.epochs:
        losses, lr_history, accuracy_history = train_autoencoder(
            model, train_loader, device, args.epochs, args.lr, args.latent_dim,
            em_only, args.checkpoint_every, CHECKPOINT_DIR, logger,
            train_files=train_files, test_files=test_files, synapse_map=synapse_map, input_size=args.input_size,
            eval_fn=eval_fn, prev_epoch=start_epoch, prev_losses=prev_losses, prev_lr_history=prev_lr_history,
            prev_accuracy_history=prev_accuracy_history, accuracy_every=1, outputs_csv_path=outputs_csv_path,
            plot_loss_acc_fn=lambda L, A: plot_loss_and_accuracy(L, A, FIG_TRAINING),
        )

    plot_loss_curve(losses, FIG_TRAINING)
    plot_lr_curve(lr_history, FIG_TRAINING)
    if accuracy_history:
        plot_loss_and_accuracy(losses, accuracy_history, FIG_TRAINING)
    plot_reconstructions(model, val_loader, device, args.epochs, FIG_TRAINING, em_only)

    embeddings, labels = extract_embeddings(model, all_files, synapse_map, device, args.input_size, em_only)
    results = evaluate_representations(embeddings, labels, logger, FIG_EVAL)
    if results.get('emb_2d') is not None:
        plot_latent_space(results['emb_2d'], labels, results['dim_method'], FIG_EVAL)

    final_acc = accuracy_history[-1] if accuracy_history else results['linear_probe_mean']
    summary_line = (
        f"AE in_ch={in_channels} latent={args.latent_dim} | "
        f"E/I acc={final_acc:.3f}  LP={results['linear_probe_mean']:.3f} 5NN={results['knn_mean']:.3f} "
        f"ARI={results['ari']:.3f} NMI={results['nmi']:.3f} Purity={results['purity']:.3f}"
    )
    summary_path = os.path.join(FIG_DIR, 'latest_run_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_line + '\n')
    logger.info("Summary: " + summary_line)


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception:
        import traceback
        err_path = os.path.join(FIG_DIR, 'run_error.log')
        os.makedirs(FIG_DIR, exist_ok=True)
        with open(err_path, 'w') as f:
            traceback.print_exc(file=f)
        raise
