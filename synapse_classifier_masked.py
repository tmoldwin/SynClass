import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'
SYNAPSE_DATA_PATH = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
BATCH_SIZE = 16
LEARNING_RATE = 0.0005  # Lower learning rate
EPOCHS = 30
INPUT_SIZE = 48
RANDOM_SEED = 42
NUM_WORKERS = 2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class MaskedSynapseDataset(Dataset):
    def __init__(self, file_list, synapse_type_map, data_dir, augment=True):
        self.file_list = file_list
        self.synapse_type_map = synapse_type_map
        self.data_dir = data_dir
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
    
    def augment_data(self, data, pre_slice, post_slice):
        """Apply data augmentation"""
        if not self.augment:
            return data, pre_slice, post_slice
            
        # Random horizontal flip
        if random.random() > 0.5:
            data = np.fliplr(data)
            pre_slice = np.fliplr(pre_slice)
            post_slice = np.fliplr(post_slice)
        
        # Random vertical flip
        if random.random() > 0.5:
            data = np.flipud(data)
            pre_slice = np.flipud(pre_slice)
            post_slice = np.flipud(post_slice)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            data = np.rot90(data, k)
            pre_slice = np.rot90(pre_slice, k)
            post_slice = np.rot90(post_slice, k)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            data = np.clip(data * factor, 0, 1)
        
        return data, pre_slice, post_slice
    
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
        
        # Apply data augmentation
        data, pre_slice, post_slice = self.augment_data(data, pre_slice, post_slice)
        
        # Proper EM data normalization
        # Use percentile-based normalization for EM data
        if data.max() > data.min():
            # Get non-zero pixels (synapse region)
            non_zero_mask = (data > 0)
            if np.sum(non_zero_mask) > 0:
                # Use 5th and 95th percentiles for robust normalization
                synapse_pixels = data[non_zero_mask]
                p5 = np.percentile(synapse_pixels, 5)
                p95 = np.percentile(synapse_pixels, 95)
                
                # Normalize using percentiles
                data = np.clip((data - p5) / (p95 - p5 + 1e-8), 0, 1)
            else:
                # If no synapse pixels, normalize globally
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            # If all values are the same, set to 0
            data = np.zeros_like(data)
        
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
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        
        # Input: 48x48 -> 24x24 -> 12x12 -> 6x6 -> 3x3
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 3, 48, 48]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)  # Add dropout after pooling
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
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
    
    # Separate E and I files
    e_files = [f for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'E']
    i_files = [f for f in valid_files if synapse_type_map[int(f.split('_')[0])] == 'I']
    
    print(f"E files: {len(e_files)}, I files: {len(i_files)}")
    
    # Balance the dataset by taking equal numbers of each class
    min_class_size = min(len(e_files), len(i_files))
    print(f"Balancing to {min_class_size} samples per class")
    
    # Randomly sample equal numbers from each class
    e_files_balanced = np.random.choice(e_files, min_class_size, replace=False)
    i_files_balanced = np.random.choice(i_files, min_class_size, replace=False)
    
    # Combine balanced files
    balanced_files = np.concatenate([e_files_balanced, i_files_balanced])
    np.random.shuffle(balanced_files)
    
    print(f"Final balanced dataset: {len(balanced_files)} files")
    print(f"E synapses: {sum(1 for f in balanced_files if synapse_type_map[int(f.split('_')[0])] == 'E')}")
    print(f"I synapses: {sum(1 for f in balanced_files if synapse_type_map[int(f.split('_')[0])] == 'I')}")
    
    # Split data
    train_files, test_files = train_test_split(balanced_files, test_size=0.2, random_state=RANDOM_SEED, 
                                              stratify=[synapse_type_map[int(f.split('_')[0])] for f in balanced_files])
    
    train_dataset = MaskedSynapseDataset(train_files, synapse_type_map, DATA_DIR, augment=True)
    test_dataset = MaskedSynapseDataset(test_files, synapse_type_map, DATA_DIR, augment=False)
    
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
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'\nConfusion Matrix (Epoch {epoch+1}):')
        print('          Predicted')
        print('          E     I')
        print(f'Actual E  {cm[0,0]:4d}  {cm[0,1]:4d}')
        print(f'      I  {cm[1,0]:4d}  {cm[1,1]:4d}')
        
        # Calculate per-class accuracy
        e_accuracy = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
        i_accuracy = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
        print(f'E class accuracy: {e_accuracy:.1f}%')
        print(f'I class accuracy: {i_accuracy:.1f}%')
        
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
    
    # Calculate class weights for balanced loss
    train_labels = []
    for file in train_dataset.file_list:
        synapse_id = int(file.split('_')[0])
        synapse_type = synapse_type_map.get(synapse_id, 'Unknown')
        label = 1 if synapse_type == 'I' else 0
        train_labels.append(label)
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: E={class_weights[0]:.3f}, I={class_weights[1]:.3f}")
    
    # Initialize model
    model = MaskedSynapseClassifier(num_classes=2).to(device)
    if args.resume and os.path.exists('best_synapse_model_masked.pth'):
        print('Resuming from best_synapse_model_masked.pth')
        model.load_state_dict(torch.load('best_synapse_model_masked.pth', map_location=device))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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