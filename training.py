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
