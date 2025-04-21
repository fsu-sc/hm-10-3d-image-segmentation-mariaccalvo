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