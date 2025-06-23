import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# Set multiprocessing start method for Windows
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'
SYNAPSE_DATA_PATH = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
BATCH_SIZE = 16  # Increased batch size for better GPU utilization
LEARNING_RATE = 0.001
EPOCHS = 50
INPUT_SIZE = 64  # Resize to 64x64xZ
RANDOM_SEED = 42
NUM_WORKERS = 4  # Number of CPU cores for data loading

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SynapseDataset(Dataset):
    def __init__(self, file_list, synapse_type_map, data_dir, transform=None):
        self.file_list = file_list
        self.synapse_type_map = synapse_type_map
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file = self.file_list[idx]
        try:
            synapse_id = int(file.split('_')[0])
        except (ValueError, IndexError):
            # Fallback for unexpected file names
            synapse_id = idx
        synapse_type = self.synapse_type_map.get(synapse_id, 'Unknown')
        
        # Load raw data
        file_path = os.path.join(self.data_dir, file)
        raw_data = np.load(file_path)
        
        # Load masks - exactly like in load_synapses.py
        pre_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'post_syn_n_mask.npy'))
        pre_mask = np.load(pre_mask_path)
        post_mask = np.load(post_mask_path)
        
        # Combine channels: [raw, pre_mask, post_mask]
        # Normalize raw data to [0, 1]
        if raw_data.max() > 0:
            raw_data = raw_data / raw_data.max()
        
        # Stack channels
        combined_data = np.stack([raw_data, pre_mask, post_mask], axis=0)  # [3, H, W, Z]
        
        # Resize to standard size (64x64xZ)
        if combined_data.shape[1] != INPUT_SIZE or combined_data.shape[2] != INPUT_SIZE:
            # Simple resize by cropping or padding
            h, w = combined_data.shape[1], combined_data.shape[2]
            if h > INPUT_SIZE:
                start_h = (h - INPUT_SIZE) // 2
                combined_data = combined_data[:, start_h:start_h+INPUT_SIZE, :, :]
            elif h < INPUT_SIZE:
                pad_h = (INPUT_SIZE - h) // 2
                combined_data = np.pad(combined_data, ((0, 0), (pad_h, INPUT_SIZE-h-pad_h), (0, 0), (0, 0)))
            
            if w > INPUT_SIZE:
                start_w = (w - INPUT_SIZE) // 2
                combined_data = combined_data[:, :, start_w:start_w+INPUT_SIZE, :]
            elif w < INPUT_SIZE:
                pad_w = (INPUT_SIZE - w) // 2
                combined_data = np.pad(combined_data, ((0, 0), (0, 0), (pad_w, INPUT_SIZE-w-pad_w), (0, 0)))
        
        # Convert to tensor
        combined_data = torch.FloatTensor(combined_data)
        
        # Label: 0 for E, 1 for I
        label = 1 if synapse_type == 'I' else 0
        
        if self.transform:
            combined_data = self.transform(combined_data)
            
        return combined_data, label, synapse_id

class SynapseClassifier3D(nn.Module):
    def __init__(self, num_classes=2):
        super(SynapseClassifier3D, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Pooling
        self.pool = nn.MaxPool3d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 3, 64, 64, Z]
        
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading synapse data...")
    
    # Load CSV data
    if not os.path.exists(SYNAPSE_DATA_PATH):
        print(f"Error: {SYNAPSE_DATA_PATH} not found!")
        return None, None, None
    
    synapse_data = pd.read_csv(SYNAPSE_DATA_PATH)
    synapse_type_map = {row['id_']: row['pre_clf_type'] for _, row in synapse_data.iterrows()}
    
    # Get all synapse files - exactly like in load_synapses.py
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('syn.npy')]
    
    # Filter files that have valid synapse types
    valid_files = []
    for file in all_files:
        try:
            synapse_id = int(file.split('_')[0])
            if synapse_id in synapse_type_map and synapse_type_map[synapse_id] in ['E', 'I']:
                valid_files.append(file)
        except (ValueError, IndexError):
            # Skip files that don't match the expected naming pattern
            continue
    
    print(f"Found {len(valid_files)} valid synapse files")
    print(f"E synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'E')}")
    print(f"I synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'I')}")
    
    if len(valid_files) == 0:
        print("No valid synapse files found!")
        return None, None, None
    
    # Split data
    train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=RANDOM_SEED, 
                                              stratify=[synapse_type_map[int(f.split('_')[0])] for f in valid_files])
    
    # Create datasets
    train_dataset = SynapseDataset(train_files, synapse_type_map, DATA_DIR)
    test_dataset = SynapseDataset(test_files, synapse_type_map, DATA_DIR)
    
    return train_dataset, test_dataset, synapse_type_map

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    """Train the model"""
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, labels, _) in enumerate(train_pbar):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with timing info
            if batch_idx % 100 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'Batch': f'{batch_idx}/{len(train_loader)}'
                })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for data, labels, _ in test_pbar:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*test_correct/test_total:.2f}%'
                })
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_synapse_model.pth')
            print(f'New best model saved! Test accuracy: {test_acc:.2f}%')
        
        print('-' * 50)
    
    return train_losses, test_losses, train_accuracies, test_accuracies, all_predictions, all_labels

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("Starting Synapse Classification Training")
    print("=" * 50)
    
    # Load data
    train_dataset, test_dataset, synapse_type_map = load_and_prepare_data()
    
    if train_dataset is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = SynapseClassifier3D(num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train model
    print("\nStarting training...")
    train_losses, test_losses, train_accuracies, test_accuracies, predictions, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS
    )
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)
    
    # Final evaluation
    print("\nFinal Results:")
    print("=" * 30)
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['E', 'I']))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 