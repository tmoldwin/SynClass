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
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'
SYNAPSE_DATA_PATH = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
INPUT_SIZE = 48  # Slightly larger for better detail
RANDOM_SEED = 42
NUM_WORKERS = 2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MaskedSynapseDataset(Dataset):
    def __init__(self, file_list, synapse_type_map, data_dir):
        self.file_list = file_list
        self.synapse_type_map = synapse_type_map
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file = self.file_list[idx]
        synapse_id = int(file.split('_')[0])
        synapse_type = self.synapse_type_map.get(synapse_id, 'Unknown')
        
        # Load raw data and masks
        file_path = os.path.join(self.data_dir, file)
        raw_data = np.load(file_path)
        
        pre_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_mask_path = os.path.join(self.data_dir, file.replace('syn.npy', 'post_syn_n_mask.npy'))
        pre_mask = np.load(pre_mask_path)
        post_mask = np.load(post_mask_path)
        
        # Create combined mask (any pixel that's part of either pre or post)
        combined_mask = np.logical_or(pre_mask, post_mask)
        
        # Apply mask to raw data - zero out non-synapse pixels
        masked_data = raw_data * combined_mask
        
        # Take the middle Z slice for 2D processing
        if masked_data.shape[2] > 0:
            z_middle = masked_data.shape[2] // 2
            data = masked_data[:, :, z_middle]
            pre_slice = pre_mask[:, :, z_middle]
            post_slice = post_mask[:, :, z_middle]
        else:
            data = masked_data[:, :, 0]
            pre_slice = pre_mask[:, :, 0]
            post_slice = post_mask[:, :, 0]
        
        # Find bounding box of synapse region
        synapse_pixels = np.where(combined_mask[:, :, z_middle if masked_data.shape[2] > 0 else 0])
        if len(synapse_pixels[0]) > 0:
            min_h, max_h = synapse_pixels[0].min(), synapse_pixels[0].max()
            min_w, max_w = synapse_pixels[1].min(), synapse_pixels[1].max()
            
            # Add padding around synapse
            padding = 8
            min_h = max(0, min_h - padding)
            max_h = min(data.shape[0], max_h + padding)
            min_w = max(0, min_w - padding)
            max_w = min(data.shape[1], max_w + padding)
            
            # Crop to synapse region
            data = data[min_h:max_h, min_w:max_w]
            pre_slice = pre_slice[min_h:max_h, min_w:max_w]
            post_slice = post_slice[min_h:max_h, min_w:max_w]
        
        # Resize to fixed size
        h, w = data.shape
        if h > INPUT_SIZE or w > INPUT_SIZE:
            # Scale down
            scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple resize by taking every nth pixel
            step_h, step_w = h // new_h, w // new_w
            data = data[::step_h, ::step_w][:new_h, :new_w]
            pre_slice = pre_slice[::step_h, ::step_w][:new_h, :new_w]
            post_slice = post_slice[::step_h, ::step_w][:new_h, :new_w]
        
        # Pad if smaller
        if data.shape[0] < INPUT_SIZE or data.shape[1] < INPUT_SIZE:
            padded_data = np.zeros((INPUT_SIZE, INPUT_SIZE))
            padded_pre = np.zeros((INPUT_SIZE, INPUT_SIZE))
            padded_post = np.zeros((INPUT_SIZE, INPUT_SIZE))
            
            h, w = data.shape
            padded_data[:h, :w] = data
            padded_pre[:h, :w] = pre_slice
            padded_post[:h, :w] = post_slice
            
            data = padded_data
            pre_slice = padded_pre
            post_slice = padded_post
        
        # Normalize data
        if data.max() > 0:
            data = data / data.max()
        
        # Stack channels: [masked_data, pre_mask, post_mask]
        combined = np.stack([data, pre_slice, post_slice], axis=0)
        
        # Convert to tensor
        combined = torch.FloatTensor(combined)
        
        # Label: 0 for E, 1 for I
        label = 1 if synapse_type == 'I' else 0
        
        return combined, label, synapse_id

class MaskedSynapseClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MaskedSynapseClassifier, self).__init__()
        
        # 2D CNN with 3 input channels (masked_data, pre_mask, post_mask)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # Input: 48x48 -> 24x24 -> 12x12 -> 6x6 -> 3x3
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 3, 48, 48]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading synapse data...")
    
    if not os.path.exists(SYNAPSE_DATA_PATH):
        print(f"Error: {SYNAPSE_DATA_PATH} not found!")
        return None, None, None
    
    synapse_data = pd.read_csv(SYNAPSE_DATA_PATH)
    synapse_type_map = {row['id_']: row['pre_clf_type'] for _, row in synapse_data.iterrows()}
    
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('syn.npy')]
    
    valid_files = []
    for file in all_files:
        try:
            synapse_id = int(file.split('_')[0])
            if synapse_id in synapse_type_map and synapse_type_map[synapse_id] in ['E', 'I']:
                valid_files.append(file)
        except (ValueError, IndexError):
            continue
    
    print(f"Found {len(valid_files)} valid synapse files")
    print(f"E synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'E')}")
    print(f"I synapses: {sum(1 for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'I')}")
    
    if len(valid_files) == 0:
        print("No valid synapse files found!")
        return None, None, None
    
    # Use more data but still manageable
    if len(valid_files) > 15000:
        print(f"Using subset of 15000 files for training")
        valid_files = np.random.choice(valid_files, 15000, replace=False)
    
    train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=RANDOM_SEED, 
                                              stratify=[synapse_type_map[int(f.split('_')[0])] for f in valid_files])
    
    train_dataset = MaskedSynapseDataset(train_files, synapse_type_map, DATA_DIR)
    test_dataset = MaskedSynapseDataset(test_files, synapse_type_map, DATA_DIR)
    
    return train_dataset, test_dataset, synapse_type_map

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
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
            
            if batch_idx % 100 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
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
        
        # Update learning rate
        scheduler.step(test_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_synapse_model_masked.pth')
            print(f'New best model saved! Test accuracy: {test_acc:.2f}%')
        
        print('-' * 50)
    
    return train_losses, test_losses, train_accuracies, test_accuracies, all_predictions, all_labels

def main():
    """Main training function"""
    print("Starting Masked Synapse Classification Training")
    print("=" * 50)
    
    train_dataset, test_dataset, synapse_type_map = load_and_prepare_data()
    
    if train_dataset is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = MaskedSynapseClassifier(num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Train model
    print("\nStarting training...")
    train_losses, test_losses, train_accuracies, test_accuracies, predictions, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS
    )
    
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