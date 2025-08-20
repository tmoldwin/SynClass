import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import DATA_DIR, CSV_PATH

# Load synapse data from CSV
synapse_data_path = CSV_PATH
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
            
            print(f'Processing synapse {synapse_id} - Type: {synapse_type}')
            
            # Check if raw data is all zeros
            if np.all(data == 0):
                print(f'Skipping synapse {synapse_id} due to all-zero raw data')
                continue
            
            # Visualize all Z slices: raw image and combined color-coded masks
            num_slices = data.shape[2]
            fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))
            fig.suptitle(f'{file} - {synapse_type}', fontsize=16)
            
            for z in range(num_slices):
                # Raw image
                axes[0, z].imshow(data[:, :, z], cmap='gray')
                axes[0, z].set_title(f'#{z+1}')
                axes[0, z].axis('off')
                
                # Combined color-coded masks: pre=pink, post=black, surround=white
                pre_slice = pre_mask[:, :, z]
                post_slice = post_mask[:, :, z]
                
                # Create RGB image for color coding
                h, w = pre_slice.shape
                rgb_mask = np.ones((h, w, 3))  # Start with white background
                
                # Pre-synaptic pixels = pink (1, 0.7, 0.7)
                pre_pixels = pre_slice.astype(bool)
                rgb_mask[pre_pixels] = [1.0, 0.7, 0.7]  # Pink
                
                # Post-synaptic pixels = black (0, 0, 0)
                post_pixels = post_slice.astype(bool)
                rgb_mask[post_pixels] = [0.0, 0.0, 0.0]  # Black
                
                # Overlapping pixels (both pre and post) = darker pink
                overlap_pixels = pre_pixels & post_pixels
                rgb_mask[overlap_pixels] = [0.8, 0.4, 0.4]  # Darker pink
                
                axes[1, z].imshow(rgb_mask)
                axes[1, z].set_title('Masks: Pre=Pink, Post=Black')
                axes[1, z].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure in the synapse demos folder
            figure_path = os.path.join(synapse_demos_dir, f'synapse_{synapse_id}_{synapse_type}.png')
            plt.savefig(figure_path)
            plt.close(fig)  # Close the figure after saving to prevent display issues
            
            # Optionally, display the figure
            # plt.show(block=False)

# Load and visualize a subset of synapse data
load_synapse_subset() 