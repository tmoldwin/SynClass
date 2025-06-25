import os
import random
import warnings
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torchvision.models as models
from torchvision.transforms import v2 as transforms

warnings.filterwarnings("ignore")

# ------------------------- configuration -------------------------
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'
CSV_PATH = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
BATCH_SIZE = 16
INPUT_XY = 224            # Standard ResNet input size
EPOCHS = 20               # 5 for feature extraction + 15 for fine-tuning
LR_HEAD = 1e-3            # Higher LR for the new classifier head
LR_FINETUNE = 1e-5        # Very low LR for fine-tuning the whole model
NUM_WORKERS = 2
RNG_SEED = 42

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='ResNet-based synapse classifier')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
args = parser.parse_args()
EPOCHS = args.epochs

# ------------------------- reproducibility ----------------------
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ------------------------- dataset ------------------------------
class Synapse2DDataset(Dataset):
    """
    Dataset class for 2D synapse classification.
    Loads synapse data, takes a 2D slice, and preprocesses it.
    """
    def __init__(self, files: List[str], type_map, data_dir, augment: bool):
        self.files = files
        self.type_map = type_map
        self.data_dir = data_dir
        self.augment = augment
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        syn_id = int(fname.split('_')[0])
        syn_type = self.type_map.get(syn_id, 'E')
        label = 1 if syn_type == 'I' else 0

        # Load arrays
        raw_path = os.path.join(self.data_dir, fname)
        pre_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'pre_syn_n_mask.npy'))
        post_path = os.path.join(self.data_dir, fname.replace('syn.npy', 'post_syn_n_mask.npy'))

        raw = np.load(raw_path)
        pre_mask = np.load(pre_path)
        post_mask = np.load(post_path)
        
        combined_mask = np.logical_or(pre_mask, post_mask)
        masked_data = raw * combined_mask

        # Take middle Z slice
        z_mid = masked_data.shape[2] // 2
        data_slice = masked_data[:, :, z_mid]
        
        # Resize to fixed size (using cv2 for better interpolation)
        data_slice = cv2.resize(data_slice, (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_AREA)

        # Percentile normalisation
        if data_slice.max() > data_slice.min():
            non_zero_mask = data_slice > 0
            if np.any(non_zero_mask):
                p5, p95 = np.percentile(data_slice[non_zero_mask], [5, 95])
                data_slice = np.clip((data_slice - p5) / (p95 - p5 + 1e-8), 0, 1)
            else: # All zero
                data_slice = np.zeros((INPUT_XY, INPUT_XY))
        
        # Convert to 3-channel image for torchvision transforms
        # The raw data is the most important channel
        image = np.stack([data_slice] * 3, axis=-1)
        image = (image * 255).astype(np.uint8) # Convert to uint8 for PIL
        
        # Apply transforms
        image_tensor = self.transform(image)

        return image_tensor, label

# ------------------------- ResNet model --------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace the final fully connected layer for our classification task
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        # Also adapt the first convolutional layer to accept 3 channels
        # This is already the default for ResNet18, so no change needed unless
        # we wanted to use the masks as separate channels. For simplicity,
        # we will use the grayscale image replicated across 3 channels.

    def forward(self, x):
        return self.resnet(x)


def main():
    # ------------------------- data preparation ---------------------
    print('Loading CSV...')
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    data = pd.read_csv(CSV_PATH)
    map_type = {r['id_']: r['pre_clf_type'] for _, r in data.iterrows()}

    all_files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('syn.npy'):
            try:
                syn_id = int(f.split('_')[0])
                if syn_id in map_type and map_type[syn_id] in ['E', 'I']:
                    all_files.append(f)
            except (ValueError, IndexError):
                continue
    E_files = [f for f in all_files if map_type[int(f.split('_')[0])] == 'E']
    I_files = [f for f in all_files if map_type[int(f.split('_')[0])] == 'I']
    size = min(len(E_files), len(I_files))
    if size == 0:
        print("Not enough data for at least one class. Exiting.")
        exit()
    random.shuffle(E_files); random.shuffle(I_files)
    files_bal = E_files[:size] + I_files[:size]
    random.shuffle(files_bal)
    train_f, test_f = train_test_split(files_bal, test_size=0.2, random_state=RNG_SEED,
                                      stratify=[map_type[int(f.split('_')[0])] for f in files_bal])

    train_ds = Synapse2DDataset(train_f, map_type, DATA_DIR, augment=True)
    val_ds   = Synapse2DDataset(test_f,  map_type, DATA_DIR, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ------------------------- training setup -----------------------
    labels_train = [1 if map_type[int(f.split('_')[0])] == 'I' else 0 for f in train_f]
    cls_w = compute_class_weight('balanced', classes=np.array([0,1]), y=labels_train)
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=device)

    model = ResNetClassifier().to(device)
    save_path = 'best_synapse_model_resnet.pth'
    if args.resume and os.path.exists(save_path):
        print(f'Resuming from checkpoint {save_path}')
        model.load_state_dict(torch.load(save_path, map_location=device))
    print('Total params:', f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=cls_w)

    # ------------------------- training loop ------------------------
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=LR_HEAD)
    
    current_best_acc = 0
    for epoch in range(1, 6): # 5 epochs for the head
        current_best_acc = run_epoch(epoch, 5, model, train_loader, val_loader, criterion, optimizer, scheduler=None, is_tuning=False, current_best_acc=current_best_acc, save_path=save_path)

    # --- STAGE 2: Fine-tune the entire model ---
    print("\n--- STAGE 2: Fine-tuning entire model ---")
    for param in model.resnet.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    for epoch in range(6, EPOCHS + 1):
        current_best_acc = run_epoch(epoch, EPOCHS, model, train_loader, val_loader, criterion, optimizer, scheduler, is_tuning=True, current_best_acc=current_best_acc, save_path=save_path)


    print(f'\nTraining complete. Best val acc: {current_best_acc:.2f}%')

def run_epoch(epoch, total_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, is_tuning, current_best_acc=0, save_path='model.pth'):
    # Training phase
    model.train()
    tot_loss = tot_corr = tot = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * x.size(0)
        tot += y.size(0)
        tot_corr += (out.argmax(1) == y).sum().item()
        pbar.set_postfix(Loss=f'{loss.item():.3f}', Acc=f'{100*tot_corr/tot:.1f}%')
    train_loss = tot_loss / tot
    train_acc = 100*tot_corr/tot

    # Validation phase
    model.eval()
    v_tot_loss = v_tot = v_corr = 0
    all_preds, all_lbls = [], []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Val]')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            v_tot_loss += loss.item() * x.size(0)
            v_tot += y.size(0)
            preds = out.argmax(1)
            v_corr += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy()); all_lbls.extend(y.cpu().numpy())
            pbar.set_postfix(Loss=f'{loss.item():.3f}', Acc=f'{100*v_corr/v_tot:.1f}%')
    val_loss = v_tot_loss / v_tot
    val_acc = 100*v_corr/v_tot
    
    if is_tuning and scheduler:
        scheduler.step(val_acc)
    
    print(f"\nEpoch {epoch}/{total_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    cm = confusion_matrix(all_lbls, all_preds)
    print('Confusion Matrix:')
    print(cm)
    if cm.shape == (2, 2):
        e_acc = cm[0,0]/cm[0].sum() if cm[0].sum() > 0 else 0
        i_acc = cm[1,1]/cm[1].sum() if cm[1].sum() > 0 else 0
        print(f'E accuracy {e_acc*100:.1f}%  |  I accuracy {i_acc*100:.1f}%')

    if epoch == total_epochs and is_tuning:
         print('\nClassification report on validation:')
         print(classification_report(all_lbls, all_preds, target_names=['E','I'], zero_division_report=0))

    return current_best_acc

if __name__ == '__main__':
    main() 