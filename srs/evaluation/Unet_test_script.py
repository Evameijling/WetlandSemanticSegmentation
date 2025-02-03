#!/usr/bin/env python3

import os
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models_015dropout import Unet
import random
import matplotlib.patches as mpatches
from torch.utils.tensorboard import SummaryWriter
import time
import rasterio
import torchmetrics
import torch.nn.functional as F
from datetime import datetime
import json
from torch.utils.data import Subset

start_time = time.time()

NUM_CLASSES = 9
BATCH_SIZE = 8
SUBSET = 100
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/10_015dropout_coscold_final_unet_model_300epochs_10subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/1_015dropout_coscold_best_unet_model_at144epochs_300epochs_1subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/10_015dropout_coscold_best_unet_model_at215epochs_300epochs_10subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/30_015dropout_coscold_best_unet_model_at200epochs_300epochs_30subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/50_015dropout_coscold_best_unet_model_at125epochs_300epochs_50subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/70_015dropout_coscold_best_unet_model_at194epochs_300epochs_70subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/nonpretrained/100_015dropout_coscold_best_unet_model_at172epochs_300epochs_100subset_nopretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/10_final_unet_model_300epochs_10subset_pretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/10_best_unet_model_at223epochs_300epochs_10subset_pretrain.pth"
MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/30_best_unet_model_at223epochs_300epochs_30subset_pretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/50_best_unet_model_at219epochs_300epochs_50subset_pretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/70_best_unet_model_at198epochs_300epochs_70subset_pretrain.pth"
# MODEL_PATH = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/inferencererun_subsets/pretrained/100_best_unet_model_at142epochs_300epochs_100subset_pretrain.pth"
SEED = 22

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Path to the JSON file with subset indices
subset_file = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/subset_indices_per_split_test.json"

# Load the subset indices
with open(subset_file, "r") as f:
    subset_indices = json.load(f)

# Choose the fraction of data to use (e.g., "10%", "50%", "100%")
fraction_key = f"{SUBSET}%"

train_indices = subset_indices["train"][fraction_key]
val_indices = subset_indices["val"][fraction_key]
test_indices = subset_indices["test"][fraction_key]

# Load data for U-net training
train_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_train_S2'
train_masks_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_train_S2'
val_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val_S2'
val_masks_dir =  '/projects/0/prjs1235/DynamicWorld_GEEData/masks_val_S2'
test_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_test_S2'
test_masks_dir =  '/projects/0/prjs1235/DynamicWorld_GEEData/masks_test_S2'

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, mask_files, transform=None, raw=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = image_files
        self.mask_files = mask_files
        self.raw = raw

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Use rasterio to read the TIFF file
        with rasterio.open(img_path) as src:
            image = src.read()  # Read all bands
            image = np.transpose(image, (1, 2, 0))  # Transpose to HWC format

        if self.raw:
            print(f"[Image {idx}] Raw: min={image.min()}, max={image.max()}")
            return image, None

        # Use rasterio to read the TIFF file for the mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read the first band (assuming mask is single-channel)

        # Clip pixel values to valid range [0, 10000]
        image = np.clip(image, 0, 10000)

        # Normalize the image
        image = image.astype(np.float32) / 10000.0

        if self.transform:
            image = self.transform(image)

        # Ensure mask is not normalized and is of type long
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Get the list of image files
train_image_files = sorted(os.listdir(train_images_dir))
train_mask_files = sorted(os.listdir(train_masks_dir))
val_image_files = sorted(os.listdir(val_images_dir))
val_mask_files = sorted(os.listdir(val_masks_dir))
test_image_files = sorted(os.listdir(test_images_dir))
test_mask_files = sorted(os.listdir(test_masks_dir))

# Create test dataset & dataloader
test_dataset = ImageMaskDataset(test_images_dir, test_masks_dir, test_image_files, test_mask_files, transform=transform)
test_dataset = Subset(test_dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print size of test set
print(f"Test set size: {len(test_dataset)}")

# Define the AverageMeter class
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Apply softmax to outputs for multi-class segmentation
        outputs = F.softmax(outputs, dim=1)

        # Flatten the tensors for batch-wise calculation
        outputs = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), 1, -1)  # Add class dim

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets, 1)

        # Compute the Dice coefficient per class
        intersection = (outputs * targets_one_hot).sum(dim=2)
        union = outputs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        # Return the average Dice Loss (1 - Dice coefficient)
        return 1 - dice_score.mean()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = Unet(num_classes=NUM_CLASSES).to(device) 

# Initialize Dice and IoU metrics using torchmetrics
dice_fn = torchmetrics.Dice(num_classes=NUM_CLASSES, average="macro").to(device)
iou_fn = torchmetrics.JaccardIndex(num_classes=NUM_CLASSES, task="multiclass", average="macro").to(device)
precision_fn = torchmetrics.Precision(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)
recall_fn = torchmetrics.Recall(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)
f1_fn = torchmetrics.F1Score(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)

def test_model(model, device, test_loader, criterion):
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    test_total_pixels = 0
    
    # Initialize metrics
    test_dice_meter = AverageMeter()
    test_iou_meter = AverageMeter()
    test_precision_meter = AverageMeter()
    test_recall_meter = AverageMeter()
    test_f1_meter = AverageMeter()

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            # Compute pixel-level accuracy
            preds = outputs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            test_corrects += (np.argmax(preds, axis=1) == masks_np).sum()
            test_total_pixels += masks_np.size

            # Compute other metrics
            dice_acc = dice_fn(outputs, masks)
            iou_acc = iou_fn(outputs, masks)

            preds_tensor = outputs.softmax(dim=1)  # shape: (B, C, H, W)
            precision = precision_fn(preds_tensor, masks)
            recall = recall_fn(preds_tensor, masks)
            f1 = f1_fn(preds_tensor, masks)

            test_dice_meter.update(dice_acc.item(), images.size(0))
            test_iou_meter.update(iou_acc.item(), images.size(0))
            test_precision_meter.update(precision.item(), images.size(0))
            test_recall_meter.update(recall.item(), images.size(0))
            test_f1_meter.update(f1.item(), images.size(0))

    # Averages / final metrics
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / test_total_pixels
    test_dice = test_dice_meter.avg
    test_iou = test_iou_meter.avg
    test_precision = test_precision_meter.avg
    test_recall = test_recall_meter.avg
    test_f1 = test_f1_meter.avg

    print(f'Test Results: Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
    print([test_acc, test_dice, test_iou, test_f1, test_precision, test_recall, test_loss])

def main():
    # 1. Load model
    print(f"Loading model from {MODEL_PATH}")
    unet.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 2. Create criterion
    criterion = DiceLoss()

    # 3. Run test
    test_model(unet, device, test_loader, criterion)

    # (Optional) Print total run time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()