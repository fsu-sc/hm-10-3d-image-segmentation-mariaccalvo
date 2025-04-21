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
