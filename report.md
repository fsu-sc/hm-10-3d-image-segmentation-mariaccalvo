# Homework Report

## Personal Details
**Name:** Maria Calvo

**Date:** 4/20/2025

**Course:** Data Science Meets Health

**Instructor:** Dr. Olmo S. Zavala Romero

Unfortunately, I was on campus for the tragic event that occured on 4/17. With also having history and also being strongly impacted by the event at Stoneman Douglas, the event has left me very shaken up. I have not been able to concentrate, it's been too soon. I'm not ready yet. I'm submitting what I was able to finish.

## Homework Questions and Answers

### Answer to 1. Dataset Access and Loading

```python
#  analyze_data.py
#  1. Dataset Access and Loading

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# loads the heart MRI dataset from the NIfTI files
data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])

# displays basic statistics
print(f"Number of training images: {len(image_files)}")
print(f"Number of training labels: {len(label_files)}")

# load a sample image and label to inspect
sample_image = os.path.join(images_dir, image_files[0])
sample_label = os.path.join(labels_dir, label_files[0])

# load using nibabel
image_nib = nib.load(sample_image)
label_nib = nib.load(sample_label)

# image and label data as numpy arrays
image = image_nib.get_fdata()
label = label_nib.get_fdata()

# shape and voxel spacing info
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Voxel spacing (pixdim): {image_nib.header.get_zooms()}")

# visualizes sample slices from different orientations 
def show_slices(img, lbl, index=None):
    if index is None:
        index = [img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # axial
    axes[0, 0].imshow(img[:, :, index[2]], cmap="gray")
    axes[0, 0].set_title("Axial - Image")
    axes[1, 0].imshow(lbl[:, :, index[2]], cmap="gray")
    axes[1, 0].set_title("Axial - Label")

    # coronal
    axes[0, 1].imshow(img[:, index[1], :], cmap="gray")
    axes[0, 1].set_title("Coronal - Image")
    axes[1, 1].imshow(lbl[:, index[1], :], cmap="gray")
    axes[1, 1].set_title("Coronal - Label")

    # sagittal
    axes[0, 2].imshow(img[index[0], :, :], cmap="gray")
    axes[0, 2].set_title("Sagittal - Image")
    axes[1, 2].imshow(lbl[index[0], :, :], cmap="gray")
    axes[1, 2].set_title("Sagittal - Label")

    # clean up axes
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# show sample slices
show_slices(image, label)

# shows the distribution of segmentation volumes
def compute_label_volume(label_data):
    return np.sum(label_data > 0)

# computes volumes for all label masks
volumes = [compute_label_volume(nib.load(os.path.join(labels_dir, f)).get_fdata()) for f in label_files]

# plot volume distribution
plt.figure(figsize=(8, 5))
plt.hist(volumes, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Segmentation Volumes")
plt.xlabel("Volume (in voxels)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# dataset for training
class HeartMRIDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # load image and label
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # normalize image to zero mean and unit variance
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        # convert to PyTorch tensors and add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)

        return {"image": image, "mask": label}

if __name__ != "__main__":
    plt.close('all')
```
Number of training images: 20
Number of training labels: 20
Image shape: (320, 320, 130)
Label shape: (320, 320, 130)
Voxel spacing (pixdim): (1.25, 1.25, 1.37)


### Answer to 2. Model Architecture

```python
#  mymodel.py
#  2. Model Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """Basic 2-layer Conv3D block: 
    -3D convolutional layers
    -3D batch normalization layers
    -ReLU activation functions"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), # 3D batch normalization
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), # 3D batch normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()

        # encoder
        # 3D max pooling layers
        self.enc1 = ConvBlock3D(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock3D(features[2], features[3])

        # decoder
        # 3D transposed conv (unsampling)
        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(features[3], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(features[2], features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(features[1], features[0])

        # final 1x1x1 conv 
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # helper to crop skip connection tensors
    def center_crop(self, enc_feature, target_shape):
        _, _, d, h, w = enc_feature.shape
        td, th, tw = target_shape
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        return enc_feature[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck(self.pool3(e3))

        # decoder with cropping for skip connection compatibility
        d3 = self.up3(b)
        e3_cropped = self.center_crop(e3, d3.shape[2:])
        d3 = torch.cat((e3_cropped, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_cropped = self.center_crop(e2, d2.shape[2:])
        d2 = torch.cat((e2_cropped, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_cropped = self.center_crop(e1, d1.shape[2:])
        d1 = torch.cat((e1_cropped, d1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# count parameters when run directly
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = UNet3D()
    print(f"Total trainable parameters: {count_parameters(model):,}")
```
Total trainable parameters: 5,602,529

### Answer to 3. Training Implementation
```python
#  training.py
#  3. Training Implementation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from mymodel import UNet3D
from analyze_data import HeartMRIDataset  # uses the Dataset class from analyze_data.py

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 

# dice loss calculation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

# crop ground truth to match model output shape
def center_crop_to_match(tensor, target):
    _, _, d, h, w = tensor.shape
    td, th, tw = target.shape[2:]

    d1 = (d - td) // 2
    h1 = (h - th) // 2
    w1 = (w - tw) // 2

    return tensor[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

# training loop
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)

        # fix: crop mask to match output shape
        masks = center_crop_to_match(masks, outputs)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# validation
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)

            # fix: crop mask to match output shape
            masks = center_crop_to_match(masks, outputs)

            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(loader)

# main training routine
def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset paths
    images_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/imagesTr"
    labels_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/labelsTr"

    # load dataset and create training and validation split
    full_dataset = HeartMRIDataset(images_dir, labels_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # model, loss, optimizer
    model = UNet3D().to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # TensorBoard setup
    writer = SummaryWriter(log_dir="runs/heart_mri_training")
    dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
    #writer.add_graph(model, dummy_input)

    # training settings
    best_val_loss = float("inf")
    patience = 5
    early_stop_counter = 0
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # log losses
        writer.add_scalars("DiceLoss", {
            "Train": train_loss,
            "Validation": val_loss
        }, epoch)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)

    writer.close()

if __name__ == "__main__":
    main()
```
Number of training images: 20
Number of training labels: 20
Image shape: (320, 320, 130)
Label shape: (320, 320, 130)
Voxel spacing (pixdim): (1.25, 1.25, 1.37)
Epoch 1/20 | Train Loss: 0.9808 | Val Loss: 0.9862
Epoch 2/20 | Train Loss: 0.9722 | Val Loss: 0.9821
Epoch 3/20 | Train Loss: 0.9696 | Val Loss: 0.9763
Epoch 4/20 | Train Loss: 0.9678 | Val Loss: 0.9851
Epoch 5/20 | Train Loss: 0.9665 | Val Loss: 0.9719
Epoch 6/20 | Train Loss: 0.9654 | Val Loss: 0.9701
Epoch 7/20 | Train Loss: 0.9650 | Val Loss: 0.9724
Epoch 8/20 | Train Loss: 0.9645 | Val Loss: 0.9693
Epoch 9/20 | Train Loss: 0.9631 | Val Loss: 0.9746
Epoch 10/20 | Train Loss: 0.9621 | Val Loss: 0.9693
Epoch 11/20 | Train Loss: 0.9616 | Val Loss: 0.9686
Epoch 12/20 | Train Loss: 0.9604 | Val Loss: 0.9741
Epoch 13/20 | Train Loss: 0.9594 | Val Loss: 0.9790
Epoch 14/20 | Train Loss: 0.9588 | Val Loss: 0.9642
Epoch 15/20 | Train Loss: 0.9575 | Val Loss: 0.9691
Epoch 16/20 | Train Loss: 0.9566 | Val Loss: 0.9884
Epoch 17/20 | Train Loss: 0.9556 | Val Loss: 0.9775
Epoch 18/20 | Train Loss: 0.9543 | Val Loss: 0.9660
Epoch 19/20 | Train Loss: 0.9534 | Val Loss: 0.9628
Epoch 20/20 | Train Loss: 0.9529 | Val Loss: 0.9599

### Answer to 4. Model Evaluation

```python
#  main.py
#  4. Model Evaluation

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader, random_split

from mymodel import UNet3D
from analyze_data import HeartMRIDataset

# data augmentation: random 90 deg rotation in one axis 
def random_rotation(image, label):
    axes = random.choice([(1, 2), (0, 2), (0, 1)])  
    k = random.randint(0, 3)  
    image = torch.rot90(image, k=k, dims=axes)
    label = torch.rot90(label, k=k, dims=axes)
    return image, label

# dice score calculation
def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# 3D visualization of results
def visualize_prediction(image, pred_mask, true_mask, title_prefix=""):
    # remove batch/channel dimensions
    image = image.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    true_mask = true_mask.squeeze().cpu().numpy()

    # choose center slices
    d, h, w = image.shape
    center = (d // 2, h // 2, w // 2)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # axial
    axes[0, 0].imshow(image[:, :, center[2]], cmap='gray')
    axes[0, 1].imshow(true_mask[:, :, center[2]], cmap='gray')
    axes[0, 2].imshow(pred_mask[:, :, center[2]], cmap='gray')

    # coronal
    axes[1, 0].imshow(image[:, center[1], :], cmap='gray')
    axes[1, 1].imshow(true_mask[:, center[1], :], cmap='gray')
    axes[1, 2].imshow(pred_mask[:, center[1], :], cmap='gray')

    # sagittal 
    axes[2, 0].imshow(image[center[0], :, :], cmap='gray')
    axes[2, 1].imshow(true_mask[center[0], :, :], cmap='gray')
    axes[2, 2].imshow(pred_mask[center[0], :, :], cmap='gray')

    for row in range(3):
        axes[row, 0].set_title(f"{title_prefix}Image")
        axes[row, 1].set_title(f"{title_prefix}Ground Truth")
        axes[row, 2].set_title(f"{title_prefix}Prediction")
        for col in range(3):
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# main evaluation script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    images_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/imagesTr"
    labels_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart/labelsTr"
    full_dataset = HeartMRIDataset(images_dir, labels_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # apply data augmentation to training dataset
    def augment_wrapper(dataset):
        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset):
                self.base = base_dataset

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                sample = self.base[idx]
                image, mask = sample['image'], sample['mask']
                image, mask = random_rotation(image, mask)
                return {'image': image, 'mask': mask}

    train_dataset = augment_wrapper(train_dataset)

    # dataloaders
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # load trained model
    model = UNet3D().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    dice_scores = []

    # evaluate on validation set
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())

            # 3D visualizations of first few cases
            if i < 3:
                visualize_prediction(images[0], outputs[0], masks[0], title_prefix=f"Case {i} - ")

    # print average dice
    avg_dice = np.mean(dice_scores)
    print(f"\nAverage Dice Score on Validation Set: {avg_dice:.3f}")

if __name__ == "__main__":
    main()
```