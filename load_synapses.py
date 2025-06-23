import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Directory containing the synapse data
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'

# Load synapse data from CSV
synapse_data_path = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
synapse_data = pd.read_csv(synapse_data_path)

# Create a mapping from synapse ID to type
synapse_type_map = {row['id_']: row['pre_clf_type'] for _, row in synapse_data.iterrows()}

# Ensure the figures directory exists
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

# Create a separate folder for synapse demos
synapse_demos_dir = os.path.join(figures_dir, 'synapse_demos')
if os.path.exists(synapse_demos_dir):
    # Clear the folder at the start of the script
    for f in os.listdir(synapse_demos_dir):
        os.remove(os.path.join(synapse_demos_dir, f))
else:
    os.makedirs(synapse_demos_dir)

# Function to load a subset of synapse data

def load_synapse_subset(num_samples=5):
    # List only the main synapse files (not mask files)
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('syn.npy')]
    
    # Randomly select a subset of files
    selected_files = np.random.choice(all_files, num_samples, replace=False)
    
    # Load and visualize the selected files
    for file in selected_files:
            synapse_id = int(file.split('_')[0])
            synapse_type = synapse_type_map.get(synapse_id, 'Unknown')
            file_path = os.path.join(DATA_DIR, file)
            data = np.load(file_path)
            
            # Correct the file path construction for pre and post synaptic masks
            pre_mask_path = os.path.join(DATA_DIR, file.replace('syn.npy', 'pre_syn_n_mask.npy'))
            post_mask_path = os.path.join(DATA_DIR, file.replace('syn.npy', 'post_syn_n_mask.npy'))
            pre_mask = np.load(pre_mask_path)
            
            post_mask = np.load(post_mask_path)
            
            # Debug: Print data shapes and contents
            print(f'Processing synapse {synapse_id} - Type: {synapse_type}')
            print(f'Raw data shape: {data.shape}')
            print(f'Pre-synaptic mask shape: {pre_mask.shape}')
            print(f'Post-synaptic mask shape: {post_mask.shape}')
            print(f'Raw data slice 1 sum: {np.sum(data[:, :, 0])}')
            print(f'Pre-synaptic mask slice 1 sum: {np.sum(pre_mask[:, :, 0])}')
            print(f'Post-synaptic mask slice 1 sum: {np.sum(post_mask[:, :, 0])}')
            
            # Detailed logging for each synapse
            print(f'--- Synapse {synapse_id} ---')
            for z in range(data.shape[2]):
                print(f'Slice {z+1}: Raw sum = {np.sum(data[:, :, z])}, Pre-synaptic sum = {np.sum(pre_mask[:, :, z])}, Post-synaptic sum = {np.sum(post_mask[:, :, z])}')
            
            # Check if raw data is all zeros
            if np.all(data == 0):
                print(f'Skipping synapse {synapse_id} due to all-zero raw data')
                continue
            
            # Visualize all Z slices in a single row with the top row numbered
            num_slices = data.shape[2]
            fig, axes = plt.subplots(3, num_slices, figsize=(4 * num_slices, 12))
            fig.suptitle(f'{file} - {synapse_type}', fontsize=16)
            
            for z in range(num_slices):
                # Raw image
                print(f'Displaying raw image for slice {z+1}')
                axes[0, z].imshow(data[:, :, z], cmap='gray')
                axes[0, z].set_title(f'#{z+1}')
                axes[0, z].axis('off')
                
                # Pre-synaptic mask
                print(f'Displaying pre-synaptic mask for slice {z+1}')
                axes[1, z].imshow(pre_mask[:, :, z], cmap='gray')
                axes[1, z].set_title('Pre')
                axes[1, z].axis('off')
                
                # Post-synaptic mask
                print(f'Displaying post-synaptic mask for slice {z+1}')
                axes[2, z].imshow(post_mask[:, :, z], cmap='gray')
                axes[2, z].set_title('Post')
                axes[2, z].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure in the synapse demos folder
            figure_path = os.path.join(synapse_demos_dir, f'synapse_{synapse_id}_{synapse_type}.png')
            plt.savefig(figure_path)
            plt.close(fig)  # Close the figure after saving to prevent display issues
            
            # Optionally, display the figure
            # plt.show(block=False)

# Load and visualize a subset of synapse data
load_synapse_subset() 