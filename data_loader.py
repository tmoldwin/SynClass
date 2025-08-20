"""
Data loading utilities for synapse classification
"""
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from constants import DATA_DIR, CSV_PATH


def load_synapse_metadata(csv_path=None):
    """Load synapse metadata from CSV file.
    
    Args:
        csv_path: Path to CSV file. If None, uses constants.CSV_PATH
        
    Returns:
        dict: Mapping from synapse ID to synapse type ('E' or 'I')
    """
    if csv_path is None:
        csv_path = CSV_PATH
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    data = pd.read_csv(csv_path)
    return {r['id_']: r['pre_clf_type'] for _, r in data.iterrows()}


def discover_synapse_files(data_dir=None, synapse_map=None):
    """Discover valid synapse files and organize by type.
    
    Args:
        data_dir: Directory containing synapse files. If None, uses constants.DATA_DIR
        synapse_map: Mapping from ID to type. If None, loads from CSV
        
    Returns:
        tuple: (E_files, I_files, synapse_map)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if synapse_map is None:
        synapse_map = load_synapse_metadata()
        
    all_files = []
    for f in os.listdir(data_dir):
        if f.endswith('syn.npy'):
            try:
                syn_id = int(f.split('_')[0])
                if syn_id in synapse_map and synapse_map[syn_id] in ['E', 'I']:
                    all_files.append(f)
            except (ValueError, IndexError):
                continue
                
    E_files = [f for f in all_files if synapse_map[int(f.split('_')[0])] == 'E']
    I_files = [f for f in all_files if synapse_map[int(f.split('_')[0])] == 'I']
    
    return E_files, I_files, synapse_map


def balance_dataset(E_files, I_files, random_seed=42):
    """Balance dataset by taking equal numbers of E and I files.
    
    Args:
        E_files: List of excitatory synapse files
        I_files: List of inhibitory synapse files
        random_seed: Random seed for shuffling
        
    Returns:
        list: Balanced list of files
    """
    size = min(len(E_files), len(I_files))
    if size == 0:
        raise ValueError("Not enough data for at least one class")
        
    random.seed(random_seed)
    random.shuffle(E_files)
    random.shuffle(I_files)
    
    files_balanced = E_files[:size] + I_files[:size]
    random.shuffle(files_balanced)
    
    return files_balanced


def create_train_test_split(files, synapse_map, test_size=0.2, random_seed=42):
    """Create stratified train/test split.
    
    Args:
        files: List of synapse files
        synapse_map: Mapping from ID to type
        test_size: Fraction of data to use for testing
        random_seed: Random seed for splitting
        
    Returns:
        tuple: (train_files, test_files)
    """
    # Create stratification labels
    labels = [synapse_map[int(f.split('_')[0])] for f in files]
    
    train_files, test_files = train_test_split(
        files,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )
    
    return train_files, test_files


def get_class_distribution(files, synapse_map):
    """Get class distribution for a list of files.
    
    Args:
        files: List of synapse files
        synapse_map: Mapping from ID to type
        
    Returns:
        dict: Dictionary with counts for each class
    """
    labels = [synapse_map[int(f.split('_')[0])] for f in files]
    return {
        'E': labels.count('E'),
        'I': labels.count('I')
    }


def prepare_synapse_data(data_dir=None, csv_path=None, test_size=0.2, random_seed=42, logger=None):
    """Complete data preparation pipeline.
    
    Args:
        data_dir: Directory containing synapse files
        csv_path: Path to CSV file with metadata
        test_size: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        logger: Logger instance for logging information
        
    Returns:
        tuple: (train_files, test_files, synapse_map, data_stats)
    """
    if logger:
        logger.info("Loading synapse metadata...")
    synapse_map = load_synapse_metadata(csv_path)
    
    if logger:
        logger.info("Discovering synapse files...")
    E_files, I_files, synapse_map = discover_synapse_files(data_dir, synapse_map)
    
    if logger:
        logger.info(f"Found {len(E_files)} E files and {len(I_files)} I files")
    
    if logger:
        logger.info("Balancing dataset...")
    balanced_files = balance_dataset(E_files, I_files, random_seed)
    
    if logger:
        logger.info(f"Using {len(balanced_files)//2} samples per class ({len(balanced_files)} total)")
    
    if logger:
        logger.info("Creating train/test split...")
    train_files, test_files = create_train_test_split(
        balanced_files, synapse_map, test_size, random_seed
    )
    
    # Get distribution stats
    train_dist = get_class_distribution(train_files, synapse_map)
    test_dist = get_class_distribution(test_files, synapse_map)
    
    data_stats = {
        'total_E': len(E_files),
        'total_I': len(I_files),
        'train_distribution': train_dist,
        'test_distribution': test_dist,
        'total_train': len(train_files),
        'total_test': len(test_files)
    }
    
    if logger:
        logger.info(f"Train: E={train_dist['E']}, I={train_dist['I']}")
        logger.info(f"Test:  E={test_dist['E']}, I={test_dist['I']}")
    
    return train_files, test_files, synapse_map, data_stats
