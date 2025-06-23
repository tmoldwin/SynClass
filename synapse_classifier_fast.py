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
BATCH_SIZE = 32  # Much larger batch size
LEARNING_RATE = 0.001
EPOCHS = 20  # Fewer epochs to test faster
INPUT_SIZE = 32  # Smaller input size for speed
RANDOM_SEED = 42
NUM_WORKERS = 2  # Reduced to avoid memory issues

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class FastSynapseDataset(Dataset):
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
        
        # Load raw data only (faster)
        file_path = os.path.join(self.data_dir, file)
        raw_data = np.load(file_path)
        
        # Simple preprocessing: take middle slice and resize
        if raw_data.shape[2] > 0:
            # Take middle Z slice
            z_middle = raw_data.shape[2] // 2
            data = raw_data[:, :, z_middle]
        else:
            data = raw_data[:, :, 0] if raw_data.shape[2] > 0 else np.zeros((raw_data.shape[0], raw_data.shape[1]))
        
        # Resize to fixed size (simple cropping)
        h, w = data.shape
        if h > INPUT_SIZE:
            start_h = (h - INPUT_SIZE) // 2
            data = data[start_h:start_h+INPUT_SIZE, :]
        if w > INPUT_SIZE:
            start_w = (w - INPUT_SIZE) // 2
            data = data[:, start_w:start_w+INPUT_SIZE]
        
        # Pad if smaller
        if h < INPUT_SIZE or w < INPUT_SIZE:
            padded = np.zeros((INPUT_SIZE, INPUT_SIZE))
            padded[:min(h, INPUT_SIZE), :min(w, INPUT_SIZE)] = data[:min(h, INPUT_SIZE), :min(w, INPUT_SIZE)]
            data = padded
        
        # Normalize
        if data.max() > 0:
            data = data / data.max()
        
        # Add channel dimension and convert to tensor
        data = torch.FloatTensor(data).unsqueeze(0)  # [1, H, W]
        
        # Label: 0 for E, 1 for I
        label = 1 if synapse_type == 'I' else 0
        
        return data, label, synapse_id

class SimpleSynapseClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleSynapseClassifier, self).__init__()
        
        # Simple 2D CNN (much faster than 3D)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        # Input: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 1, 32, 32]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
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
    
    # Use smaller subset for faster testing
    if len(valid_files) > 5000:
        print(f"Using subset of 5000 files for faster training")
        valid_files = np.random.choice(valid_files, 5000, replace=False)
    
    train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=RANDOM_SEED, 
                                              stratify=[synapse_type_map[int(f.split('_')[0])] for f in valid_files])
    
    train_dataset = FastSynapseDataset(train_files, synapse_type_map, DATA_DIR)
    test_dataset = FastSynapseDataset(test_files, synapse_type_map, DATA_DIR)
    
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
            
            if batch_idx % 50 == 0:
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
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_synapse_model_fast.pth')
            print(f'New best model saved! Test accuracy: {test_acc:.2f}%')
        
        print('-' * 50)
    
    return train_losses, test_losses, train_accuracies, test_accuracies, all_predictions, all_labels

def main():
    """Main training function"""
    print("Starting Fast Synapse Classification Training")
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
    model = SimpleSynapseClassifier(num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Train model
    print("\nStarting training...")
    train_losses, test_losses, train_accuracies, test_accuracies, predictions, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS
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