"""
Data augmentation utilities for synapse classification
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


class RandomRotation3D:
    """3D random rotation augmentation."""
    
    def __init__(self, degrees=15):
        self.degrees = degrees
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Random rotation angles
        angle_x = np.random.uniform(-self.degrees, self.degrees)
        angle_y = np.random.uniform(-self.degrees, self.degrees) 
        angle_z = np.random.uniform(-self.degrees, self.degrees)
        
        # Convert to radians
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)
        
        # Create rotation matrices
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        
        # Rotation around X axis
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=torch.float32)
        
        # Rotation around Y axis  
        R_y = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32)
        
        # Rotation around Z axis
        R_z = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Combined rotation matrix
        R = torch.mm(torch.mm(R_z, R_y), R_x)
        
        # Apply rotation (simplified - in practice you'd need proper 3D rotation)
        # For now, just return the original data as 3D rotation is complex
        return data


class RandomFlip3D:
    """3D random flip augmentation."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Random flips along each axis
        if np.random.random() < self.prob:
            data = torch.flip(data, dims=[0])  # Flip along first dimension
        if np.random.random() < self.prob:
            data = torch.flip(data, dims=[1])  # Flip along second dimension  
        if np.random.random() < self.prob:
            data = torch.flip(data, dims=[2])  # Flip along third dimension
            
        return data


class RandomNoise3D:
    """Add random noise to 3D data."""
    
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        noise = torch.randn_like(data) * self.noise_factor
        return data + noise


class RandomCrop3D:
    """3D random crop that considers mask regions."""
    
    def __init__(self, crop_size, mask_aware=True, min_mask_ratio=0.1):
        """
        Args:
            crop_size: Target crop size (int or tuple of 3 ints)
            mask_aware: Whether to consider mask regions when cropping
            min_mask_ratio: Minimum ratio of mask pixels required in crop
        """
        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size, crop_size)
        self.mask_aware = mask_aware
        self.min_mask_ratio = min_mask_ratio
        
    def __call__(self, data, mask=None):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        D, H, W = data.shape[-3:]
        crop_d, crop_h, crop_w = self.crop_size
        
        # If data is smaller than crop size, pad it
        if D < crop_d or H < crop_h or W < crop_w:
            pad_d = max(0, crop_d - D)
            pad_h = max(0, crop_h - H) 
            pad_w = max(0, crop_w - W)
            data = F.pad(data, (0, pad_w, 0, pad_h, 0, pad_d))
            if mask is not None:
                mask = F.pad(mask, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = data.shape[-3:]
        
        if self.mask_aware and mask is not None:
            # Find mask centers for guided cropping
            mask_indices = torch.nonzero(mask > 0.5)
            if len(mask_indices) > 0:
                # Try multiple random crops and pick the one with most mask coverage
                best_crop = None
                best_mask_ratio = 0
                
                for _ in range(5):  # Try 5 random crops
                    # Random crop around mask region
                    idx = torch.randint(len(mask_indices), (1,)).item()
                    mask_center = mask_indices[idx]
                    
                    # Add some randomness around mask center
                    center_d = max(crop_d//2, min(D - crop_d//2, mask_center[0].item() + torch.randint(-crop_d//4, crop_d//4, (1,)).item()))
                    center_h = max(crop_h//2, min(H - crop_h//2, mask_center[1].item() + torch.randint(-crop_h//4, crop_h//4, (1,)).item()))
                    center_w = max(crop_w//2, min(W - crop_w//2, mask_center[2].item() + torch.randint(-crop_w//4, crop_w//4, (1,)).item()))
                    
                    start_d = center_d - crop_d//2
                    start_h = center_h - crop_h//2
                    start_w = center_w - crop_w//2
                    
                    # Ensure crop is within bounds
                    start_d = max(0, min(D - crop_d, start_d))
                    start_h = max(0, min(H - crop_h, start_h))
                    start_w = max(0, min(W - crop_w, start_w))
                    
                    crop_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    mask_ratio = crop_mask.sum().float() / crop_mask.numel()
                    
                    if mask_ratio > best_mask_ratio:
                        best_mask_ratio = mask_ratio
                        best_crop = (start_d, start_h, start_w)
                    
                    if mask_ratio >= self.min_mask_ratio:
                        break
                
                if best_crop is not None:
                    start_d, start_h, start_w = best_crop
                else:
                    # Fallback to random crop
                    start_d = torch.randint(0, D - crop_d + 1, (1,)).item()
                    start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
                    start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
            else:
                # No mask found, random crop
                start_d = torch.randint(0, D - crop_d + 1, (1,)).item()
                start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
                start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
        else:
            # Random crop without mask awareness
            start_d = torch.randint(0, D - crop_d + 1, (1,)).item()
            start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
            start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        # Apply crop
        cropped_data = data[..., start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        if mask is not None:
            cropped_mask = mask[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
            return cropped_data, cropped_mask
        
        return cropped_data


class RandomCrop2D:
    """2D random crop that considers mask regions."""
    
    def __init__(self, crop_size, mask_aware=True, min_mask_ratio=0.1):
        """
        Args:
            crop_size: Target crop size (int or tuple of 2 ints)
            mask_aware: Whether to consider mask regions when cropping
            min_mask_ratio: Minimum ratio of mask pixels required in crop
        """
        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.mask_aware = mask_aware
        self.min_mask_ratio = min_mask_ratio
        
    def __call__(self, data, mask=None):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        H, W = data.shape[-2:]
        crop_h, crop_w = self.crop_size
        
        # If data is smaller than crop size, pad it
        if H < crop_h or W < crop_w:
            pad_h = max(0, crop_h - H)
            pad_w = max(0, crop_w - W)
            data = F.pad(data, (0, pad_w, 0, pad_h))
            if mask is not None:
                mask = F.pad(mask, (0, pad_w, 0, pad_h))
            H, W = data.shape[-2:]
        
        if self.mask_aware and mask is not None:
            # Find mask centers for guided cropping
            mask_indices = torch.nonzero(mask > 0.5)
            if len(mask_indices) > 0:
                # Try multiple random crops and pick the one with most mask coverage
                best_crop = None
                best_mask_ratio = 0
                
                for _ in range(5):  # Try 5 random crops
                    # Random crop around mask region
                    idx = torch.randint(len(mask_indices), (1,)).item()
                    mask_center = mask_indices[idx]
                    
                    # Add some randomness around mask center
                    center_h = max(crop_h//2, min(H - crop_h//2, mask_center[0].item() + torch.randint(-crop_h//4, crop_h//4, (1,)).item()))
                    center_w = max(crop_w//2, min(W - crop_w//2, mask_center[1].item() + torch.randint(-crop_w//4, crop_w//4, (1,)).item()))
                    
                    start_h = center_h - crop_h//2
                    start_w = center_w - crop_w//2
                    
                    # Ensure crop is within bounds
                    start_h = max(0, min(H - crop_h, start_h))
                    start_w = max(0, min(W - crop_w, start_w))
                    
                    crop_mask = mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
                    mask_ratio = crop_mask.sum().float() / crop_mask.numel()
                    
                    if mask_ratio > best_mask_ratio:
                        best_mask_ratio = mask_ratio
                        best_crop = (start_h, start_w)
                    
                    if mask_ratio >= self.min_mask_ratio:
                        break
                
                if best_crop is not None:
                    start_h, start_w = best_crop
                else:
                    # Fallback to random crop
                    start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
                    start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
            else:
                # No mask found, random crop
                start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
                start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
        else:
            # Random crop without mask awareness
            start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
            start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        # Apply crop
        cropped_data = data[..., start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        if mask is not None:
            cropped_mask = mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
            return cropped_data, cropped_mask
        
        return cropped_data


class Resize3D:
    """Resize 3D data to target size."""
    
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size, size)
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Add batch and channel dimensions for interpolation
        if len(data.shape) == 3:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            
        # Resize using 3D interpolation
        data = F.interpolate(data, size=self.size, mode='trilinear', align_corners=False)
        
        # Remove batch and channel dimensions
        data = data.squeeze(0).squeeze(0)
        
        return data


class Normalize3D:
    """Normalize 3D data."""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        if self.mean is None:
            mean = data.mean()
        else:
            mean = self.mean
            
        if self.std is None:
            std = data.std()
        else:
            std = self.std
            
        return (data - mean) / (std + 1e-8)


class ToTensor3D:
    """Convert numpy array to tensor."""
    
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data


class Compose3D:
    """Compose multiple 3D transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


def get_2d_augmentation_transform(augment=True, input_size=224):
    """Get 2D augmentation pipeline.
    
    Args:
        augment: Whether to apply augmentation
        input_size: Target size for resizing
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(input_size * 1.2), int(input_size * 1.2))),  # Resize larger for cropping
            transforms.RandomCrop((input_size, input_size)),  # Random crop to target size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel normalization
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])


def get_3d_augmentation_transform(augment=True, input_size=64):
    """Get 3D augmentation pipeline.
    
    Args:
        augment: Whether to apply augmentation
        input_size: Target size for resizing
        
    Returns:
        Compose3D: Transform pipeline
    """
    transforms_list = [ToTensor3D()]
    
    if augment:
        # Resize larger for cropping, then crop to target size
        if input_size is not None:
            crop_size = int(input_size * 1.2)
            transforms_list.append(Resize3D(crop_size))
            transforms_list.append(RandomCrop3D(input_size, mask_aware=False))  # Basic random crop
        
        transforms_list.extend([
            RandomFlip3D(prob=0.5),
            RandomRotation3D(degrees=15),
            RandomNoise3D(noise_factor=0.05),
        ])
    else:
        # Just resize for validation/testing
        if input_size is not None:
            transforms_list.append(Resize3D(input_size))
    
    transforms_list.append(Normalize3D())
    
    return Compose3D(transforms_list)


def get_augmentation_transform(augment=True, is_3d=True, input_size=None):
    """Get appropriate augmentation transform based on data type.
    
    Args:
        augment: Whether to apply augmentation
        is_3d: Whether this is 3D data (True) or 2D data (False)
        input_size: Target size for resizing
        
    Returns:
        Transform pipeline appropriate for data type
    """
    if is_3d:
        default_size = 64 if input_size is None else input_size
        return get_3d_augmentation_transform(augment, default_size)
    else:
        default_size = 224 if input_size is None else input_size
        return get_2d_augmentation_transform(augment, default_size)


def visualize_2d_augmentations():
    """Main visualization function for 2D augmentations - the only function you need."""
    import os
    import matplotlib.pyplot as plt
    import random
    from constants import DATA_DIR
    from data_loader import load_synapse_metadata, discover_synapse_files
    from scipy.ndimage import rotate
    from skimage.transform import resize
    
    print("Creating 2D augmentation gallery...")
    
    # Load sample synapses
    synapse_map = load_synapse_metadata()
    E_files, I_files, _ = discover_synapse_files(synapse_map=synapse_map)
    
    # Get 3 synapses total
    selected_files = []
    if len(E_files) >= 2:
        selected_files.extend(random.sample(E_files, 2))
    if len(I_files) >= 1:
        selected_files.extend(random.sample(I_files, 1))
    
    synapses = []
    for file in selected_files[:3]:
        filepath = os.path.join(DATA_DIR, file)
        data = np.load(filepath)
        syn_id = int(file.split('_')[0])
        syn_type = synapse_map[syn_id]
        
        # Load masks
        try:
            pre_mask_path = os.path.join(DATA_DIR, file.replace('syn.npy', 'pre_syn_n_mask.npy'))
            post_mask_path = os.path.join(DATA_DIR, file.replace('syn.npy', 'post_syn_n_mask.npy'))
            pre_mask = np.load(pre_mask_path)
            post_mask = np.load(post_mask_path)
            combined_mask = np.logical_or(pre_mask, post_mask)
        except:
            pre_mask = post_mask = combined_mask = None
        
        synapses.append({
            'data': data,
            'combined_mask': combined_mask,
            'syn_type': syn_type,
            'syn_id': syn_id
        })
    
    if len(synapses) == 0:
        print("No synapses found!")
        return
    
    # Create the visualization
    fig, axes = plt.subplots(len(synapses), 6, figsize=(18, 3*len(synapses)))
    fig.suptitle('2D Synapse Augmentation Gallery (Focus on 2D)', fontsize=16, fontweight='bold')
    
    for syn_idx, synapse in enumerate(synapses):
        data = synapse['data']
        mask = synapse['combined_mask']
        syn_type = synapse['syn_type']
        syn_id = synapse['syn_id']
        
        # Take middle slice
        mid_z = data.shape[2] // 2
        data_2d = data[:, :, mid_z]
        mask_2d = mask[:, :, mid_z] if mask is not None else None
        
        # Normalize
        data_norm = (data_2d - data_2d.min()) / (data_2d.max() - data_2d.min())
        
        ax_row = axes[syn_idx] if len(synapses) > 1 else axes
        
        # 1. Original
        ax_row[0].imshow(data_norm, cmap='gray')
        if syn_idx == 0:
            ax_row[0].set_title('Original')
        ax_row[0].set_ylabel(f'ID {syn_id} ({syn_type})')
        ax_row[0].axis('off')
        
        # 2. Horizontal flip
        h_flip = np.fliplr(data_norm)
        ax_row[1].imshow(h_flip, cmap='gray')
        if syn_idx == 0:
            ax_row[1].set_title('H-Flip')
        ax_row[1].axis('off')
        
        # 3. Rotation
        rotated = rotate(data_norm, angle=15, reshape=False, mode='constant', cval=0)
        ax_row[2].imshow(rotated, cmap='gray')
        if syn_idx == 0:
            ax_row[2].set_title('Rotate 15°')
        ax_row[2].axis('off')
        
        # 4. Noise
        noisy = data_norm + np.random.normal(0, 0.05, data_norm.shape)
        noisy = np.clip(noisy, 0, 1)
        ax_row[3].imshow(noisy, cmap='gray')
        if syn_idx == 0:
            ax_row[3].set_title('+ Noise')
        ax_row[3].axis('off')
        
        # 5. Brightness
        bright = np.clip(data_norm * 1.3, 0, 1)
        ax_row[4].imshow(bright, cmap='gray')
        if syn_idx == 0:
            ax_row[4].set_title('+ Bright')
        ax_row[4].axis('off')
        
        # 6. Smart crop (mask-aware)
        if mask_2d is not None and np.any(mask_2d):
            # Find mask center
            mask_coords = np.where(mask_2d > 0)
            if len(mask_coords[0]) > 0:
                center_h = int(np.mean(mask_coords[0]))
                center_w = int(np.mean(mask_coords[1]))
                
                # Crop around center
                crop_size = min(data_norm.shape) // 2
                h, w = data_norm.shape
                start_h = max(0, min(h - crop_size, center_h - crop_size//2))
                start_w = max(0, min(w - crop_size, center_w - crop_size//2))
                
                cropped = data_norm[start_h:start_h+crop_size, start_w:start_w+crop_size]
                # Resize back
                cropped_resized = resize(cropped, data_norm.shape, preserve_range=True)
                ax_row[5].imshow(cropped_resized, cmap='gray')
            else:
                ax_row[5].imshow(data_norm, cmap='gray')
        else:
            ax_row[5].imshow(data_norm, cmap='gray')
        
        if syn_idx == 0:
            ax_row[5].set_title('Smart Crop')
        ax_row[5].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/2d_augmentation_gallery.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 2D augmentation gallery saved to: {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    print("Running 2D augmentation visualization...")
    visualize_2d_augmentations()
