import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random

from DeepGeol.deepgeol.unet import UNet


# Dataset 
class NPYDataset(Dataset):
    def __init__(self, data_path, mask_path, augment=False):
        self.data = np.load(data_path)
        self.masks = np.load(mask_path)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.masks[idx]

        # augmentation
        if self.augment:
            if random.random() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()

        x = torch.tensor(x).float().permute(2, 0, 1)

        y = torch.tensor(y).float()
        if y.ndim == 3:
            y = y.permute(2, 0, 1)
        else:
            y = y.unsqueeze(0)

        return x, y


# Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# Paths
data_path = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/training_data.npy"
mask_path = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/training_masks.npy"


# Dataset, Split
full_dataset = NPYDataset(data_path, mask_path, augment=False)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

train_dataset = NPYDataset(data_path, mask_path, augment=True)
train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)

val_dataset = torch.utils.data.Subset(full_dataset, val_subset.indices)


# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# Model
model = UNet(input_channels=3).to(device)

# Loss
bce = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0]).to(device)
)

def criterion(pred, y):
    return bce(pred, y) + dice_loss(pred, y)


# Optimizer 
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-4
)

epochs = 50

# Scheduler (OneCycle)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)

# Early stopping  
patience = 5
best_val_loss = float("inf")
counter = 0

os.makedirs("checkpoints", exist_ok=True)

# Training loop
train_losses = []
val_losses = []

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            val_loss += loss.item()


    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    
    print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "checkpoints/best_model_v2.pth")
        print(" Best model saved")

    else:
        counter += 1
        print(f" No improvement ({counter}/{patience})")

    if counter >= patience:
        print(" Early stopping")
        break