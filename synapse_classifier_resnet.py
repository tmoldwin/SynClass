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

warnings.filterwarnings("ignore")

# ------------------------- configuration -------------------------
DATA_DIR = 'Data/synpase_raw_em/synpase_raw_em/'
CSV_PATH = 'Data/synpase_raw_em/synpase_raw_em/synapse_data.csv'
BATCH_SIZE = 8            # Smaller batch size for larger images
INPUT_XY = 224            # Standard ResNet input size
EPOCHS = 30
LR = 1e-4                 # Smaller LR for fine-tuning
NUM_WORKERS = 2
RNG_SEED = 42

# ------------------------- argparse ------------------------------
parser = argparse.ArgumentParser(description='ResNet-based synapse classifier')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=EPOCHS)
parser.add_argument('--lr', type=float, default=LR)
args = parser.parse_args()
EPOCHS = args.epochs
LR = args.lr

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

    def __len__(self):
        return len(self.files)

    def _augment(self, data, pre_slice, post_slice):
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

        return data.copy(), pre_slice.copy(), post_slice.copy()

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
        pre_slice = pre_mask[:, :, z_mid]
        post_slice = post_mask[:, :, z_mid]
        
        # Bounding box and crop (this part is less important with large resize, but can help focus)
        synapse_pixels = np.where(combined_mask[:, :, z_mid])
        if len(synapse_pixels[0]) > 0:
            min_h, max_h = synapse_pixels[0].min(), synapse_pixels[0].max()
            min_w, max_w = synapse_pixels[1].min(), synapse_pixels[1].max()
            
            padding = 8
            min_h = max(0, min_h - padding)
            max_h = min(data_slice.shape[0], max_h + padding + 1)
            min_w = max(0, min_w - padding)
            max_w = min(data_slice.shape[1], max_w + padding + 1)
            
            data_slice = data_slice[min_h:max_h, min_w:max_w]
            pre_slice = pre_slice[min_h:max_h, min_w:max_w]
            post_slice = post_slice[min_h:max_h, min_w:max_w]
        
        # Resize to fixed size (using cv2 for better interpolation)
        data_slice = cv2.resize(data_slice, (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_AREA)
        pre_slice = cv2.resize(pre_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)
        post_slice = cv2.resize(post_slice.astype(float), (INPUT_XY, INPUT_XY), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        data_slice, pre_slice, post_slice = self._augment(data_slice, pre_slice, post_slice)

        # Percentile normalisation
        if data_slice.max() > data_slice.min():
            non_zero_mask = data_slice > 0
            if np.any(non_zero_mask):
                p5, p95 = np.percentile(data_slice[non_zero_mask], [5, 95])
                data_slice = np.clip((data_slice - p5) / (p95 - p5 + 1e-8), 0, 1)
            else: # All zero
                data_slice = np.zeros((INPUT_XY, INPUT_XY))
        
        image = np.stack([data_slice, pre_slice, post_slice], axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        return image, label

# ------------------------- ResNet model --------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer for our classification task
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

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
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # ------------------------- training loop ------------------------
    best_acc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tot_loss = tot_corr = tot = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
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

        # ---- validation ----
        model.eval()
        v_tot_loss = v_tot = v_corr = 0
        all_preds, all_lbls = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]')
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
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        cm = confusion_matrix(all_lbls, all_preds)
        print('Confusion Matrix:')
        print(cm)
        if cm.shape == (2, 2):
            e_acc = cm[0,0]/cm[0].sum() if cm[0].sum() > 0 else 0
            i_acc = cm[1,1]/cm[1].sum() if cm[1].sum() > 0 else 0
            print(f'E accuracy {e_acc*100:.1f}%  |  I accuracy {i_acc*100:.1f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'New best saved ({best_acc:.2f}%)')

    print(f'\nTraining complete. Best val acc: {best_acc:.2f}%')

    print('\nClassification report on validation:')
    print(classification_report(all_lbls, all_preds, target_names=['E','I'], zero_division_report=0))

if __name__ == '__main__':
    main() 