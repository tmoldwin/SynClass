#!/usr/bin/env python3
"""
Script to create a mini dataset with 100 synapses from the full dataset.
This creates a new folder 'synapses_raw_em_mini' with selected synapse files and CSV.
"""

import os
import shutil
import pandas as pd
import random
from pathlib import Path

def create_mini_dataset(source_dir='Data/synpase_raw_em', 
                       target_dir='synapses_raw_em_mini',
                       num_synapses=100,
                       random_seed=42):
    """
    Create a mini dataset by selecting random synapses from the full dataset.
    
    Args:
        source_dir: Source directory containing synapse files and CSV
        target_dir: Target directory for mini dataset
        num_synapses: Number of synapses to include (default: 100)
        random_seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    print(f"Creating mini dataset in: {target_path}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Number of synapses: {num_synapses}")
    
    # Load the full CSV
    csv_path = Path(source_dir) / 'synapse_data.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total synapses in full dataset: {len(df)}")
    print(f"Synapse types: {df['pre_clf_type'].value_counts().to_dict()}")
    
    # Randomly sample synapses
    sampled_df = df.sample(n=num_synapses, random_state=random_seed)
    
    # Ensure we have a good balance of E and I synapses
    e_count = (sampled_df['pre_clf_type'] == 'E').sum()
    i_count = (sampled_df['pre_clf_type'] == 'I').sum()
    print(f"Selected {e_count} excitatory and {i_count} inhibitory synapses")
    
    # Save the mini CSV
    mini_csv_path = target_path / 'synapse_data.csv'
    sampled_df.to_csv(mini_csv_path, index=False)
    print(f"Saved mini CSV to: {mini_csv_path}")
    
    # Copy synapse files for selected synapses
    synapse_ids = sampled_df['id_'].tolist()
    print(f"Copying files for {len(synapse_ids)} synapses...")
    
    copied_count = 0
    missing_count = 0
    
    for syn_id in synapse_ids:
        # Define the three file types for each synapse
        file_patterns = [
            f"{syn_id}_syn.npy",
            f"{syn_id}_pre_syn_n_mask.npy", 
            f"{syn_id}_post_syn_n_mask.npy"
        ]
        
        for pattern in file_patterns:
            source_file = Path(source_dir) / pattern
            target_file = target_path / pattern
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copied_count += 1
            else:
                print(f"Warning: File not found: {source_file}")
                missing_count += 1
    
    print(f"Copying complete!")
    print(f"Files copied: {copied_count}")
    print(f"Files missing: {missing_count}")
    
    # Verify the mini dataset
    print("\nVerifying mini dataset...")
    
    # Check CSV
    mini_df = pd.read_csv(mini_csv_path)
    print(f"Mini CSV has {len(mini_df)} rows")
    
    # Check files
    mini_files = list(target_path.glob("*.npy"))
    print(f"Mini dataset has {len(mini_files)} .npy files")
    
    # Count unique synapses in files
    unique_synapses = set()
    for file_path in mini_files:
        syn_id = file_path.stem.split('_')[0]
        unique_synapses.add(syn_id)
    
    print(f"Mini dataset covers {len(unique_synapses)} unique synapses")
    
    # Update constants.py to point to the mini dataset
    update_constants_for_mini_dataset(target_dir)
    
    print(f"\nMini dataset created successfully!")
    print(f"To use it, update your data loading code to point to: {target_dir}")
    print(f"Or use the updated constants.py that now points to the mini dataset")

def update_constants_for_mini_dataset(mini_dir):
    """Update constants.py to point to the mini dataset"""
    
    constants_path = 'constants.py'
    if not os.path.exists(constants_path):
        print("Warning: constants.py not found, skipping update")
        return
    
    # Read current constants
    with open(constants_path, 'r') as f:
        content = f.read()
    
    # Update the data directory paths
    new_content = content.replace(
        "DATA_DIR = 'Data/synpase_raw_em/'",
        f"DATA_DIR = '{mini_dir}/'"
    )
    new_content = new_content.replace(
        "CSV_PATH = 'Data/synpase_raw_em/synapse_data.csv'",
        f"CSV_PATH = '{mini_dir}/synapse_data.csv'"
    )
    
    # Write updated constants
    with open(constants_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated constants.py to point to mini dataset: {mini_dir}")

if __name__ == "__main__":
    # Create mini dataset with 100 synapses
    create_mini_dataset()
