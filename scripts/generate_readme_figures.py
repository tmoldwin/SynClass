#!/usr/bin/env python3
"""
Generate figures for the SynClass GitHub README.
Run from project root: python scripts/generate_readme_figures.py

Produces:
- figures/data_examples.png: Raw EM + mask overlays
- figures/comprehensive_2d_augmentation_gallery.png: Augmented examples
- figures/architecture_diagram.png: CNN architecture
"""
import os
import sys

# Run from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def ensure_data():
    """Ensure synapse data exists; generate synthetic if needed."""
    from constants import DATA_DIR, DATA_ARCHIVE, CSV_PATH, CSV_PATH_FALLBACK
    has_csv = os.path.isfile(CSV_PATH) or os.path.isfile(CSV_PATH_FALLBACK)
    has_archive = os.path.isfile(DATA_ARCHIVE)
    has_dir = False
    if os.path.isdir(DATA_DIR):
        try:
            has_dir = any(f.endswith('syn.npy') for f in os.listdir(DATA_DIR))
        except OSError:
            pass
    if has_csv and (has_archive or has_dir):
        return True
    print("No real data found. Generating synthetic data for figure generation...")
    from generate_synthetic_data import generate_synthetic_dataset
    generate_synthetic_dataset(output_dir=DATA_DIR, num_samples=50)
    return True


def generate_data_examples():
    """Create data examples figure: raw EM + mask overlays."""
    import matplotlib.pyplot as plt
    import numpy as np
    from constants import DATA_DIR, DATA_ARCHIVE
    from data_loader import load_synapse_metadata, discover_synapse_files

    try:
        from data_utils import load_npy
        archive_path = DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None
    except ImportError:
        load_npy = None
        archive_path = None

    def _load(filename):
        path = os.path.join(DATA_DIR, filename)
        if load_npy and (archive_path or not os.path.isfile(path)):
            return load_npy(DATA_DIR, archive_path or DATA_ARCHIVE, filename)
        return np.load(path)

    synapse_map = load_synapse_metadata()
    E_files, I_files, _ = discover_synapse_files(synapse_map=synapse_map)
    import random
    selected = (random.sample(E_files, min(2, len(E_files))) if E_files else []) + \
               (random.sample(I_files, min(2, len(I_files))) if I_files else [])
    selected = selected[:4]

    if not selected:
        print("No synapse files; skipping data_examples.png")
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Synapse Data Examples (EM volume middle slice)', fontsize=14, fontweight='bold')

    for i, file in enumerate(selected):
        data = _load(file)
        pre_mask = _load(file.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_mask = _load(file.replace('syn.npy', 'post_syn_n_mask.npy'))
        syn_id = int(file.split('_')[0])
        syn_type = synapse_map[syn_id]
        mid_z = data.shape[2] // 2
        img = data[:, :, mid_z]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        pre_2d = pre_mask[:, :, mid_z]
        post_2d = post_mask[:, :, mid_z]

        axes[i, 0].imshow(img_norm, cmap='gray')
        axes[i, 0].set_title(f'Raw EM (ID {syn_id}, {syn_type})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_norm, cmap='gray')
        h, w = img_norm.shape
        pre_overlay = np.zeros((h, w, 4))
        post_overlay = np.zeros((h, w, 4))
        pre_overlay[pre_2d.astype(bool)] = [1.0, 1.0, 0.0, 0.5]
        post_overlay[post_2d.astype(bool)] = [1.0, 0.4, 0.7, 0.5]
        axes[i, 1].imshow(pre_overlay)
        axes[i, 1].imshow(post_overlay)
        axes[i, 1].set_title('Masks (Yellow=Pre, Pink=Post)')
        axes[i, 1].axis('off')

        rgb = np.ones((h, w, 3))
        rgb[pre_2d.astype(bool)] = [1.0, 0.7, 0.7]
        rgb[post_2d.astype(bool)] = [0.3, 0.3, 0.3]
        axes[i, 2].imshow(rgb)
        axes[i, 2].set_title('Masks only')
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/data_examples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/data_examples.png")


def generate_augmentation_gallery():
    """Run augmentation gallery (uses archive if available)."""
    try:
        from augmentation import visualize_2d_augmentations
        visualize_2d_augmentations()
        if os.path.isfile('figures/comprehensive_2d_augmentation_gallery.png'):
            print("Saved figures/comprehensive_2d_augmentation_gallery.png")
    except ImportError as e:
        print(f"Skipping augmentation gallery (missing dep: {e}). Install scikit-image for full figures.")


def generate_architecture_diagram():
    """Create architecture diagram for CNN2DMultiChannel (depth=3)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def box(x, y, w, h, label, color='#e8f4f8'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', facecolor=color, edgecolor='#333')
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, wrap=True)

    # Input (top)
    box(3.5, 10.2, 3, 0.8, 'Input: 3 ch (EM + Pre mask + Post mask)\n224 x 224', '#d4edda')

    # Conv blocks
    box(2.5, 8.2, 5, 1.2, 'Conv2d(3->64, 7x7, s=2) + BN + ReLU + MaxPool', '#fff3cd')
    box(2.5, 6.2, 5, 1.2, 'Conv2d(64->128, 5x5) + BN + ReLU + MaxPool', '#fff3cd')
    box(2.5, 4.2, 5, 1.2, 'Conv2d(128->256, 3x3) + BN + ReLU + AdaptiveAvgPool', '#fff3cd')

    # Classifier
    box(2.5, 1.8, 5, 2.0, 'Flatten -> Linear(256->256) -> BN -> ReLU -> Dropout(0.3)\n-> Linear(256->64) -> BN -> ReLU -> Linear(64->2)', '#e2d5f1')

    # Output (bottom)
    box(3.5, 0.2, 3, 1.2, 'Output: 2 classes\nE (excitatory) / I (inhibitory)', '#f8d7da')

    # Arrows (head at lower y)
    for (y1, y2) in [(10.2, 9.4), (8.2, 7.4), (6.2, 5.4), (4.2, 3.8), (1.8, 1.4)]:
        ax.annotate('', xy=(5, y2), xytext=(5, y1), arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_title('CNN2DMultiChannel Architecture (depth=3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/architecture_diagram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/architecture_diagram.png")


def main():
    ensure_data()
    generate_data_examples()
    generate_augmentation_gallery()
    generate_architecture_diagram()
    print("\nDone. Run training for accuracy curves: python synapse_classifier_2dcnn_multichannel.py --epochs 50")


if __name__ == '__main__':
    main()
