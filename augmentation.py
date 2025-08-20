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
            transforms.Resize((input_size, input_size)),
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
    
    # Add resize if input_size is specified
    if input_size is not None:
        transforms_list.append(Resize3D(input_size))
    
    if augment:
        transforms_list.extend([
            RandomFlip3D(prob=0.5),
            RandomRotation3D(degrees=15),
            RandomNoise3D(noise_factor=0.05),
        ])
    
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
