"""
Generic preprocessing for synapse classification.
Extract middle slice, augment, resize, normalize to multi-channel tensor.
Reusable across different models (2D CNN, ResNet, etc.).
"""
import os
import numpy as np
import torch
from torchvision.transforms import functional as TF

try:
    from data_utils import load_npy
except ImportError:
    load_npy = None


def rotated_crop(image, center_h, center_w, crop_size, angle):
    """Apply rotated crop to image without black borders.
    
    Args:
        image: 2D numpy array (H, W)
        center_h, center_w: Crop center
        crop_size: Output crop size
        angle: Rotation angle in degrees
        
    Returns:
        Cropped 2D array
    """
    import cv2
    h, w = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((float(center_w), float(center_h)), float(angle), 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    start_h = center_h - crop_size // 2
    start_w = center_w - crop_size // 2
    end_h = start_h + crop_size
    end_w = start_w + crop_size
    start_h = max(0, start_h)
    start_w = max(0, start_w)
    end_h = min(h, end_h)
    end_w = min(w, end_w)
    cropped = rotated[start_h:end_h, start_w:end_w]
    if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
        pad_h = max(0, crop_size - cropped.shape[0])
        pad_w = max(0, crop_size - cropped.shape[1])
        cropped = cv2.copyMakeBorder(cropped, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return cropped


def preprocess_synapse_2d(data_3d, pre_mask_3d, post_mask_3d, input_size=224, augment=False, seed=None):
    """Convert 3D synapse volume + masks to 2D masked-image tensor.
    
    Output: (2, H, W) tensor with channels [EM*pre_mask, EM*post_mask].
    Only pixels within each mask region are visible; background is zero.
    
    Returns:
        torch.Tensor of shape (2, input_size, input_size)
    """
    import cv2
    z_data = data_3d.shape[2]
    z_pre = pre_mask_3d.shape[2]
    z_post = post_mask_3d.shape[2]
    # Random z-slice when augmenting (training); fixed middle slice otherwise (val)
    if augment and seed is not None:
        np.random.seed(seed % (2**31))
        z_ix = np.random.randint(0, z_data) if z_data > 1 else 0
        z_pre_ix = min(z_ix, z_pre - 1) if z_pre > 1 else 0
        z_post_ix = min(z_ix, z_post - 1) if z_post > 1 else 0
    else:
        z_ix = z_data // 2
        z_pre_ix = (z_pre // 2) if z_pre > 1 else 0
        z_post_ix = (z_post // 2) if z_post > 1 else 0
    data_2d = data_3d[:, :, z_ix]
    pre_mask_2d = pre_mask_3d[:, :, z_pre_ix]
    post_mask_2d = post_mask_3d[:, :, z_post_ix]

    data_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-8)
    data_uint8 = (data_norm * 255).astype(np.uint8)
    pre_uint8 = (pre_mask_2d * 255).astype(np.uint8)
    post_uint8 = (post_mask_2d * 255).astype(np.uint8)

    if augment and seed is not None:
        np.random.seed(seed % (2**31))
        if np.random.random() > 0.5:
            data_uint8 = np.flipud(data_uint8)
            pre_uint8 = np.flipud(pre_uint8)
            post_uint8 = np.flipud(post_uint8)
        if np.random.random() > 0.5:
            data_uint8 = np.fliplr(data_uint8)
            pre_uint8 = np.fliplr(pre_uint8)
            post_uint8 = np.fliplr(post_uint8)
        h, w = data_uint8.shape
        crop_ratio = np.random.uniform(0.65, 0.80)
        crop_size = int(min(h, w) * crop_ratio)
        angle = np.random.uniform(-45, 45)
        combined_mask = (pre_mask_2d > 0) | (post_mask_2d > 0)
        if np.any(combined_mask):
            mask_coords = np.where(combined_mask)
            center_h = int(np.mean(mask_coords[0]))
            center_w = int(np.mean(mask_coords[1]))
            offset_h = np.random.randint(-h//8, h//8)
            offset_w = np.random.randint(-w//8, w//8)
            center_h = np.clip(center_h + offset_h, crop_size//2, h - crop_size//2)
            center_w = np.clip(center_w + offset_w, crop_size//2, w - crop_size//2)
        else:
            center_h = np.random.randint(crop_size//2, h - crop_size//2)
            center_w = np.random.randint(crop_size//2, w - crop_size//2)
        data_uint8 = rotated_crop(data_uint8, center_h, center_w, crop_size, angle)
        pre_uint8 = rotated_crop(pre_uint8, center_h, center_w, crop_size, angle)
        post_uint8 = rotated_crop(post_uint8, center_h, center_w, crop_size, angle)
        noise = np.random.normal(0, 0.03 * 255, data_uint8.shape).astype(np.int16)
        data_uint8 = np.clip(data_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    data_resized = cv2.resize(data_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)
    pre_resized = cv2.resize(pre_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    post_resized = cv2.resize(post_uint8, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

    data_f = data_resized.astype(np.float32) / 255.0
    pre_f = pre_resized.astype(np.float32) / 255.0
    post_f = post_resized.astype(np.float32) / 255.0

    masked_pre = torch.tensor(data_f * (pre_f > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
    masked_post = torch.tensor(data_f * (post_f > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
    masked_pre = TF.normalize(masked_pre, mean=[0.485], std=[0.229])
    masked_post = TF.normalize(masked_post, mean=[0.485], std=[0.229])
    return torch.cat([masked_pre, masked_post], dim=0)


def preprocess_synapse_multislice(data_3d, pre_mask_3d, post_mask_3d, n_slices, input_size=224, augment=False, seed=None, use_masks=False):
    """Convert 3D volume to 2D tensor with N randomly sampled z-slices as channels.

    use_masks=False: N channels (EM only, one per slice). Shape (N, H, W).
    use_masks=True:  3*N channels (EM, pre, post per slice). Shape (3*N, H, W).
    Same augmentation (flips, crop, noise) is applied consistently across all slices.
    """
    import cv2
    z_data = data_3d.shape[2]
    z_pre = pre_mask_3d.shape[2]
    z_post = post_mask_3d.shape[2]
    z_data = max(1, z_data)
    if augment and seed is not None:
        np.random.seed(seed % (2**31))
        z_indices = np.random.randint(0, z_data, size=n_slices) if z_data > 1 else np.zeros(n_slices, dtype=np.int64)
    else:
        if z_data >= n_slices:
            z_indices = np.linspace(0, z_data - 1, n_slices, dtype=np.int64)
        else:
            z_indices = np.minimum(np.arange(n_slices) % z_data, z_data - 1)

    channels = []
    for z_ix in z_indices:
        z_pre_ix = min(int(z_ix), z_pre - 1) if z_pre > 1 else 0
        z_post_ix = min(int(z_ix), z_post - 1) if z_post > 1 else 0
        data_2d = data_3d[:, :, int(z_ix)]
        data_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-8)
        data_uint8 = (data_norm * 255).astype(np.uint8)
        if use_masks:
            pre_uint8 = (pre_mask_3d[:, :, z_pre_ix] * 255).astype(np.uint8)
            post_uint8 = (post_mask_3d[:, :, z_post_ix] * 255).astype(np.uint8)
            channels.append((data_uint8, pre_uint8, post_uint8))
        else:
            channels.append((data_uint8,))

    if augment and seed is not None:
        np.random.seed(seed % (2**31))
        flip_ud = np.random.random() > 0.5
        flip_lr = np.random.random() > 0.5
        h, w = channels[0][0].shape
        crop_ratio = np.random.uniform(0.65, 0.80)
        crop_size = int(min(h, w) * crop_ratio)
        angle = np.random.uniform(-45, 45)
        if use_masks and np.any((channels[0][1].astype(np.float32) + channels[0][2].astype(np.float32)) > 0):
            combined = (channels[0][1].astype(np.float32) + channels[0][2].astype(np.float32)) > 0
            mask_coords = np.where(combined)
            center_h = int(np.mean(mask_coords[0]))
            center_w = int(np.mean(mask_coords[1]))
            center_h = np.clip(center_h + np.random.randint(-h//8, h//8), crop_size//2, h - crop_size//2)
            center_w = np.clip(center_w + np.random.randint(-w//8, w//8), crop_size//2, w - crop_size//2)
        else:
            center_h = np.random.randint(crop_size//2, h - crop_size//2)
            center_w = np.random.randint(crop_size//2, w - crop_size//2)
        for i in range(len(channels)):
            if use_masks:
                d, p, q = channels[i]
                if flip_ud:
                    d, p, q = np.flipud(d), np.flipud(p), np.flipud(q)
                if flip_lr:
                    d, p, q = np.fliplr(d), np.fliplr(p), np.fliplr(q)
                d = rotated_crop(d, center_h, center_w, crop_size, angle)
                p = rotated_crop(p, center_h, center_w, crop_size, angle)
                q = rotated_crop(q, center_h, center_w, crop_size, angle)
                noise = np.random.normal(0, 0.03 * 255, d.shape).astype(np.int16)
                d = np.clip(d.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                channels[i] = (d, p, q)
            else:
                (d,) = channels[i]
                if flip_ud:
                    d = np.flipud(d)
                if flip_lr:
                    d = np.fliplr(d)
                d = rotated_crop(d, center_h, center_w, crop_size, angle)
                noise = np.random.normal(0, 0.03 * 255, d.shape).astype(np.int16)
                d = np.clip(d.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                channels[i] = (d,)

    tensors = []
    for c in channels:
        data_uint8 = c[0]
        data_resized = cv2.resize(data_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)
        data_t = torch.tensor(data_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        data_t = TF.normalize(data_t, mean=[0.485], std=[0.229])
        tensors.append(data_t)
        if use_masks:
            pre_resized = cv2.resize(c[1], (input_size, input_size), interpolation=cv2.INTER_NEAREST)
            post_resized = cv2.resize(c[2], (input_size, input_size), interpolation=cv2.INTER_NEAREST)
            tensors.append(torch.tensor(pre_resized, dtype=torch.float32).unsqueeze(0) / 255.0)
            tensors.append(torch.tensor(post_resized, dtype=torch.float32).unsqueeze(0) / 255.0)
    return torch.cat(tensors, dim=0)


def load_synapse_data(data_dir, archive_path, filename):
    """Load synapse volume and masks from disk or archive."""
    filepath = os.path.join(data_dir, filename)
    if load_npy and (archive_path or not os.path.isfile(filepath)):
        data_3d = load_npy(data_dir, archive_path, filename)
    else:
        data_3d = np.load(filepath)
    pre_name = filename.replace('syn.npy', 'pre_syn_n_mask.npy')
    post_name = filename.replace('syn.npy', 'post_syn_n_mask.npy')
    try:
        if load_npy and (archive_path or not os.path.isfile(os.path.join(data_dir, pre_name))):
            pre_mask_3d = load_npy(data_dir, archive_path, pre_name)
            post_mask_3d = load_npy(data_dir, archive_path, post_name)
        else:
            pre_mask_3d = np.load(os.path.join(data_dir, pre_name))
            post_mask_3d = np.load(os.path.join(data_dir, post_name))
    except FileNotFoundError:
        pre_mask_3d = np.zeros_like(data_3d)
        post_mask_3d = np.zeros_like(data_3d)
    return data_3d, pre_mask_3d, post_mask_3d


def create_balanced_epoch_indices(file_list, synapse_map, examples_per_epoch):
    """Create balanced class indices for sampling."""
    labels = []
    for filename in file_list:
        syn_id = int(filename.split('_')[0])
        labels.append(synapse_map[syn_id])
    e_indices = [i for i, label in enumerate(labels) if label == 'E']
    i_indices = [i for i, label in enumerate(labels) if label == 'I']
    samples_per_class = examples_per_epoch // 2
    np.random.seed(42)
    selected_e = np.random.choice(e_indices, min(samples_per_class, len(e_indices)), replace=False)
    selected_i = np.random.choice(i_indices, min(samples_per_class, len(i_indices)), replace=False)
    all_selected = np.concatenate([selected_e, selected_i])
    np.random.shuffle(all_selected)
    return all_selected[:examples_per_epoch]


class MultiChannelSynapseDataset(torch.utils.data.Dataset):
    """2D dataset with 3 channels: image + pre_mask + post_mask.
    
    Generic dataset using preprocessing.preprocess_synapse_2d.
    Reusable across different models.
    """
    def __init__(self, file_list, synapse_map, data_dir=None, archive_path=None, augment=False,
                 input_size=224, examples_per_epoch=None, augment_prob=0.2):
        from constants import DATA_DIR, DATA_ARCHIVE
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.archive_path = archive_path if archive_path and os.path.isfile(archive_path) else (DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None)
        self.augment = augment
        self.input_size = input_size
        self.examples_per_epoch = examples_per_epoch if examples_per_epoch is not None else len(file_list)
        self.augment_prob = augment_prob
        self._epoch_indices = None
        print(f"Dataset: {len(file_list)} total synapses -> {self.examples_per_epoch} examples per epoch")
        if augment:
            print(f"Augmentation probability: {augment_prob:.1f} (vs {1-augment_prob:.1f} for originals)")

    def __len__(self):
        return self.examples_per_epoch

    def __getitem__(self, idx):
        if self.examples_per_epoch >= len(self.file_list):
            synapse_idx = idx % len(self.file_list)
            aug_idx = idx // len(self.file_list)
        else:
            if self._epoch_indices is None:
                self._epoch_indices = create_balanced_epoch_indices(
                    self.file_list, self.synapse_map, self.examples_per_epoch
                )
                n_e = sum(1 for i in self._epoch_indices if self.synapse_map[int(self.file_list[i].split('_')[0])] == 'E')
                print(f"Balanced sampling: {n_e} E + {len(self._epoch_indices)-n_e} I = {len(self._epoch_indices)} total examples")
            synapse_idx = self._epoch_indices[idx]
            aug_idx = 0

        filename = self.file_list[synapse_idx]
        filepath = os.path.join(self.data_dir, filename)
        for attempt in range(3):
            try:
                data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
                    self.data_dir, self.archive_path, filename
                )
                if data_3d.size == 0:
                    raise ValueError("empty array")
                break
            except (FileNotFoundError, EOFError, OSError, ValueError):
                if attempt < 2:
                    synapse_idx = (synapse_idx + 1) % len(self.file_list)
                    filename = self.file_list[synapse_idx]
                else:
                    raise

        syn_id = int(filename.split('_')[0])
        syn_type = self.synapse_map[syn_id]
        label = 1 if syn_type == 'I' else 0
        seed = hash((synapse_idx, aug_idx)) % (2**31) if self.augment else None
        multi_channel = preprocess_synapse_2d(
            data_3d, pre_mask_3d, post_mask_3d,
            input_size=self.input_size,
            augment=self.augment,
            seed=seed
        )
        return multi_channel, label


class MultiSliceSynapseDataset(torch.utils.data.Dataset):
    """Dataset that samples N z-slices per volume. use_masks=False -> N channels (EM only); use_masks=True -> 3*N channels."""
    def __init__(self, file_list, synapse_map, n_slices=3, use_masks=False, data_dir=None, archive_path=None, augment=False,
                 input_size=224, examples_per_epoch=None):
        from constants import DATA_DIR, DATA_ARCHIVE
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.n_slices = n_slices
        self.use_masks = use_masks
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.archive_path = archive_path if archive_path and os.path.isfile(archive_path) else (DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None)
        self.augment = augment
        self.input_size = input_size
        self.examples_per_epoch = examples_per_epoch if examples_per_epoch is not None else len(file_list)
        self._epoch_indices = None
        ch = 3 * n_slices if use_masks else n_slices
        print(f"Dataset: {len(file_list)} total -> {self.examples_per_epoch} per epoch, n_slices={n_slices} use_masks={use_masks} -> {ch} channels")

    def __len__(self):
        return self.examples_per_epoch

    def __getitem__(self, idx):
        if self.examples_per_epoch >= len(self.file_list):
            synapse_idx = idx % len(self.file_list)
            aug_idx = idx // len(self.file_list)
        else:
            if self._epoch_indices is None:
                self._epoch_indices = create_balanced_epoch_indices(
                    self.file_list, self.synapse_map, self.examples_per_epoch
                )
            synapse_idx = self._epoch_indices[idx]
            aug_idx = 0
        filename = self.file_list[synapse_idx]
        for attempt in range(3):
            try:
                data_3d, pre_mask_3d, post_mask_3d = load_synapse_data(
                    self.data_dir, self.archive_path, filename
                )
                if data_3d.size == 0:
                    raise ValueError("empty array")
                break
            except (FileNotFoundError, EOFError, OSError, ValueError):
                if attempt < 2:
                    synapse_idx = (synapse_idx + 1) % len(self.file_list)
                    filename = self.file_list[synapse_idx]
                else:
                    raise
        syn_id = int(filename.split('_')[0])
        label = 1 if self.synapse_map[syn_id] == 'I' else 0
        seed = hash((synapse_idx, aug_idx)) % (2**31) if self.augment else None
        tensor = preprocess_synapse_multislice(
            data_3d, pre_mask_3d, post_mask_3d, self.n_slices,
            input_size=self.input_size, augment=self.augment, seed=seed, use_masks=self.use_masks
        )
        return tensor, label


def create_multislice_dataloaders(train_files, test_files, synapse_map, n_slices=3, use_masks=False,
                                  batch_size=16, num_workers=4, pin_memory=True,
                                  input_size=224, augment_train=True, examples_per_epoch=None,
                                  augment_val=False, val_examples_per_epoch=None,
                                  data_dir=None):
    """Create dataloaders for N-slice. use_masks=False -> N channels (EM only); use_masks=True -> 3*N channels."""
    from torch.utils.data import DataLoader
    from constants import DATA_ARCHIVE
    if data_dir:
        archive_path = None
    else:
        archive_path = DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None
    if archive_path and num_workers > 0:
        num_workers = 0
    train_dataset = MultiSliceSynapseDataset(
        train_files, synapse_map, n_slices=n_slices, use_masks=use_masks, data_dir=data_dir, archive_path=archive_path,
        augment=augment_train, input_size=input_size, examples_per_epoch=examples_per_epoch
    )
    val_dataset = MultiSliceSynapseDataset(
        test_files, synapse_map, n_slices=n_slices, use_masks=use_masks, data_dir=data_dir, archive_path=archive_path,
        augment=augment_val, input_size=input_size, examples_per_epoch=val_examples_per_epoch
    )
    ch = 3 * n_slices if use_masks else n_slices
    print(f"TRAINING CONFIGURATION (multislice): n_slices={n_slices}, use_masks={use_masks}, channels={ch}")
    print(f"   Training: {len(train_files)} -> {len(train_dataset)} examples per epoch")
    print(f"   Validation: {len(test_files)} -> {len(val_dataset)} examples per epoch")
    prefetch = 4 if num_workers > 0 else None  # multislice aug is heavy, prefetch more
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch
    )
    return train_loader, val_loader


def create_multichannel_dataloaders(train_files, test_files, synapse_map,
                                   batch_size=16, num_workers=4, pin_memory=True,
                                   input_size=224, augment_train=True, examples_per_epoch=None,
                                   augment_val=False, val_examples_per_epoch=None,
                                   data_dir=None):
    """Create dataloaders for multi-channel training. Generic for any model.
    If data_dir is set (e.g. Data/proofread_synapses), archive is not used."""
    from torch.utils.data import DataLoader
    from constants import DATA_ARCHIVE
    if data_dir:
        archive_path = None
    else:
        archive_path = DATA_ARCHIVE if os.path.isfile(DATA_ARCHIVE) else None
    if archive_path and num_workers > 0:
        num_workers = 0
    train_dataset = MultiChannelSynapseDataset(
        train_files, synapse_map, data_dir=data_dir, archive_path=archive_path, augment=augment_train,
        input_size=input_size, examples_per_epoch=examples_per_epoch
    )
    val_dataset = MultiChannelSynapseDataset(
        test_files, synapse_map, data_dir=data_dir, archive_path=archive_path, augment=augment_val,
        input_size=input_size, examples_per_epoch=val_examples_per_epoch
    )
    print(f"TRAINING CONFIGURATION:")
    print(f"   Training: {len(train_files)} total synapses -> {len(train_dataset)} examples per epoch")
    print(f"   Validation: {len(test_files)} total synapses -> {len(val_dataset)} examples per epoch")
    print(f"   Training/Val ratio: {len(train_dataset)/len(val_dataset):.1f}:1")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader
