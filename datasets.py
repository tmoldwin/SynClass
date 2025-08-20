"""
Dataset classes for synapse classification
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentation import get_augmentation_transform, RandomCrop3D, RandomCrop2D
from constants import DATA_DIR


class SynapseDataset(Dataset):
    """Base dataset class for synapse data.
    
    Args:
        file_list: List of synapse files
        synapse_map: Mapping from synapse ID to type
        data_dir: Directory containing synapse files
        augment: Whether to apply data augmentation
        is_3d: Whether this is 3D data (True) or 2D data (False)
        input_size: Target size for resizing (for 2D: int, for 3D: tuple or None)
        use_mask_aware_crop: Whether to use mask-aware cropping
    """
    
    def __init__(self, file_list, synapse_map, data_dir=None, augment=False, 
                 is_3d=True, input_size=None, use_mask_aware_crop=False):
        self.file_list = file_list
        self.synapse_map = synapse_map
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.augment = augment
        self.is_3d = is_3d
        self.input_size = input_size
        self.use_mask_aware_crop = use_mask_aware_crop
        
        # Get augmentation transform
        self.transform = get_augmentation_transform(
            augment=augment, 
            is_3d=is_3d, 
            input_size=input_size
        )
        
        # Setup mask-aware cropping if needed
        if use_mask_aware_crop and augment:
            if is_3d:
                self.mask_crop = RandomCrop3D(input_size, mask_aware=True, min_mask_ratio=0.15)
            else:
                self.mask_crop = RandomCrop2D(input_size, mask_aware=True, min_mask_ratio=0.15)
        else:
            self.mask_crop = None
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # Load synapse data
        data = np.load(filepath)
        
        # Get label
        syn_id = int(filename.split('_')[0])
        syn_type = self.synapse_map[syn_id]
        label = 1 if syn_type == 'I' else 0  # I=1 (inhibitory), E=0 (excitatory)
        
        # Load masks if using mask-aware cropping
        mask = None
        if self.mask_crop is not None:
            try:
                pre_mask_path = os.path.join(self.data_dir, filename.replace('syn.npy', 'pre_syn_n_mask.npy'))
                post_mask_path = os.path.join(self.data_dir, filename.replace('syn.npy', 'post_syn_n_mask.npy'))
                pre_mask = np.load(pre_mask_path)
                post_mask = np.load(post_mask_path)
                # Combine pre and post masks
                mask = np.logical_or(pre_mask, post_mask).astype(np.float32)
                mask = torch.tensor(mask, dtype=torch.float32)
            except FileNotFoundError:
                # If masks not found, continue without mask-aware cropping
                pass
        
        # Convert data to tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Apply mask-aware cropping first if available
        if self.mask_crop is not None and mask is not None:
            if mask.dim() == 3 and data.dim() == 3:  # 3D case
                data, _ = self.mask_crop(data, mask)
            elif mask.dim() == 2 and data.dim() == 2:  # 2D case
                data, _ = self.mask_crop(data, mask)
        
        # Apply other transforms
        if self.transform:
            data = self.transform(data)
        
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        return data, torch.tensor(label, dtype=torch.long)


class Synapse3DDataset(SynapseDataset):
    """3D synapse dataset with 3D-specific preprocessing."""
    
    def __init__(self, file_list, synapse_map, data_dir=None, augment=False, input_size=64, use_mask_aware_crop=True):
        super().__init__(
            file_list=file_list,
            synapse_map=synapse_map, 
            data_dir=data_dir,
            augment=augment,
            is_3d=True,
            input_size=input_size,
            use_mask_aware_crop=use_mask_aware_crop
        )


class Synapse2DDataset(SynapseDataset):
    """2D synapse dataset with 2D-specific preprocessing."""
    
    def __init__(self, file_list, synapse_map, data_dir=None, augment=False, input_size=224, use_mask_aware_crop=True):
        super().__init__(
            file_list=file_list,
            synapse_map=synapse_map,
            data_dir=data_dir, 
            augment=augment,
            is_3d=False,
            input_size=input_size,
            use_mask_aware_crop=use_mask_aware_crop
        )
    
    def __getitem__(self, idx):
        # Get 3D data first
        data, label = super().__getitem__(idx)
        
        # Convert 3D to 2D by taking middle slice or max projection
        if len(data.shape) == 3:
            # Take middle slice along the first dimension
            middle_idx = data.shape[0] // 2
            data = data[middle_idx]
        elif len(data.shape) == 4 and data.shape[0] == 1:
            # Remove batch dimension and take middle slice
            data = data[0]
            middle_idx = data.shape[0] // 2
            data = data[middle_idx]
            
        # Ensure 2D data has channel dimension for CNN
        if len(data.shape) == 2:
            data = data.unsqueeze(0)  # Add channel dimension
            
        return data, label


def create_dataloaders(train_files, test_files, synapse_map, batch_size=16, 
                      num_workers=4, pin_memory=True, is_3d=True, 
                      input_size=None, augment_train=True, use_mask_aware_crop=True):
    """Create train and validation dataloaders.
    
    Args:
        train_files: List of training files
        test_files: List of test files
        synapse_map: Mapping from synapse ID to type
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        is_3d: Whether to use 3D or 2D dataset
        input_size: Input size for resizing
        augment_train: Whether to augment training data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Choose dataset class
    DatasetClass = Synapse3DDataset if is_3d else Synapse2DDataset
    
    # Create datasets
    train_dataset = DatasetClass(
        train_files, synapse_map, augment=augment_train, input_size=input_size, 
        use_mask_aware_crop=use_mask_aware_crop
    )
    val_dataset = DatasetClass(
        test_files, synapse_map, augment=False, input_size=input_size,
        use_mask_aware_crop=False  # No cropping for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader
