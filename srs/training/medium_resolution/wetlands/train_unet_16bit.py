# https://youtu.be/hTpq9lzAb8M
"""
Train U-net by loading pre-trained encoder weights.
"""

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
# from models import Unet
# from models_shallow2block import Unet
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

############################# SETTING UP THE PARAMETERS #############################

SIZE = 256
BATCH_SIZE = 8
EPOCHS = 300
NUM_CLASSES = 9
NUM_IMAGES = None
SUBSET = 100
SEED = 42
HYPERPAR = '100'

config = 'pretrain'
# config = 'nopretrain'

############################# SETTING UP THE SEED #############################

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

############################# LOADING THE SUBSET INDICES #############################

# Path to the JSON file with subset indices
subset_file = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/subset_indices_per_split.json"

# Load the subset indices
with open(subset_file, "r") as f:
    subset_indices = json.load(f)

# Choose the fraction of data to use (e.g., "10%", "50%", "100%")
fraction_key = f"{SUBSET}%"

train_indices = subset_indices["train"][fraction_key]
val_indices = subset_indices["val"][fraction_key]
test_indices = subset_indices["test"][fraction_key]


############################# LOADING THE DATA #############################

# Ensure the saved_images directory exists
output_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/saved_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure the models directory exists
models_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load data for U-net training
train_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_train_S2'
train_masks_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_train_S2'
val_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val_S2'
val_masks_dir =  '/projects/0/prjs1235/DynamicWorld_GEEData/masks_val_S2'
test_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_test_S2'
test_masks_dir =  '/projects/0/prjs1235/DynamicWorld_GEEData/masks_test_S2'

# Custom Dataset
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
train_image_files = sorted(os.listdir(train_images_dir))[:NUM_IMAGES]
train_mask_files = sorted(os.listdir(train_masks_dir))[:NUM_IMAGES]
val_image_files = sorted(os.listdir(val_images_dir))[:NUM_IMAGES]
val_mask_files = sorted(os.listdir(val_masks_dir))[:NUM_IMAGES]
test_image_files = sorted(os.listdir(test_images_dir))[:NUM_IMAGES]
test_mask_files = sorted(os.listdir(test_masks_dir))[:NUM_IMAGES]

print(f"Number of training images for U-Net: {len(train_image_files)}")
print(f"Number of validation images for U-Net: {len(val_image_files)}")
print(f"Number of test images for U-Net: {len(test_image_files)}")

train_dataset = ImageMaskDataset(train_images_dir, train_masks_dir, train_image_files, train_mask_files, transform=transform)
val_dataset = ImageMaskDataset(val_images_dir, val_masks_dir, val_image_files, val_mask_files, transform=transform)
test_dataset = ImageMaskDataset(test_images_dir, test_masks_dir, test_image_files, test_mask_files, transform=transform)
raw_dataset = ImageMaskDataset(test_images_dir, test_masks_dir, test_image_files, test_mask_files, raw=True)

# Create subsets using the loaded indices
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)
test_dataset = Subset(test_dataset, test_indices)
raw_dataset = Subset(raw_dataset, test_indices)

# Print the number of images in each subset
print(f"Number of training images in subset: {len(train_dataset)}")
print(f"Number of validation images in subset: {len(val_dataset)}")
print(f"Number of test images in subset: {len(test_dataset)}")

# Create DataLoaders for subsets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

############################# DEFINE TRAINING LOSS AND FUNCTIONS #############################

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
    
# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = Unet(num_classes=NUM_CLASSES).to(device)  # Ensure the model is initialized with the correct number of classes
criterion = DiceLoss()  
optimizer = optim.Adam(unet.parameters(), lr=0.001, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize TensorBoard writer with date and time in the run name
# log_dir = f'/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/runs/train_unet_{config}_{current_time}'
log_dir = f'/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/runs/train_unet_{config}_{HYPERPAR}'
writer = SummaryWriter(log_dir=log_dir)

# Load pre-trained encoder weights
if config == 'pretrain':
    pretrained_encoder_weights = torch.load('/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/Autoencoder_for_pretraining/100percent/05_03_02_00001_015dropout_final_autoencoder_encoderweights_200epochs.pth', weights_only=True) 
    unet.encoder.load_state_dict(pretrained_encoder_weights)

# Initialize Dice and IoU metrics using torchmetrics
dice_fn = torchmetrics.Dice(num_classes=NUM_CLASSES, average="macro").to(device)
iou_fn = torchmetrics.JaccardIndex(num_classes=NUM_CLASSES, task="multiclass", average="macro").to(device)
precision_fn = torchmetrics.Precision(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)
recall_fn = torchmetrics.Recall(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)
f1_fn = torchmetrics.F1Score(num_classes=NUM_CLASSES, average='macro', task="multiclass").to(device)

# Initialize AverageMeter objects for running and validation metrics
running_dice_meter = AverageMeter()
running_iou_meter = AverageMeter()
running_precision_meter = AverageMeter()
running_recall_meter = AverageMeter()
running_f1_meter = AverageMeter()

val_dice_meter = AverageMeter()
val_iou_meter = AverageMeter()
val_precision_meter = AverageMeter()
val_recall_meter = AverageMeter()
val_f1_meter = AverageMeter()

# Calculate IoU for a single image
def calculate_iou(pred, target, num_classes=NUM_CLASSES):
    iou_list = []
    pred = pred.flatten()
    target = target.flatten()
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth or prediction, do not include in IoU calculation
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

############################# TESTING FUNCTION (SAVES/LOADS) ################################
def test_model_checkpoint(checkpoint_path):
    # Load the checkpoint model
    unet.load_state_dict(torch.load(checkpoint_path))
    unet.eval()

    test_loss = 0.0
    test_corrects = 0
    test_dice_meter = AverageMeter()
    test_iou_meter = AverageMeter()
    test_precision_meter = AverageMeter()
    test_recall_meter = AverageMeter()
    test_f1_meter = AverageMeter()
    test_total_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = unet(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = outputs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            test_corrects += (np.argmax(preds, axis=1) == masks_np).sum()
            test_total_pixels += masks_np.size

            # Calculate metrics
            dice_acc = dice_fn(outputs, masks)
            iou_acc = iou_fn(outputs, masks)

            preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)
            precision = precision_fn(preds_tensor, masks)
            recall = recall_fn(preds_tensor, masks)
            f1 = f1_fn(preds_tensor, masks)

            test_dice_meter.update(dice_acc.item(), images.size(0))
            test_iou_meter.update(iou_acc.item(), images.size(0))
            test_precision_meter.update(precision.item(), images.size(0))
            test_recall_meter.update(recall.item(), images.size(0))
            test_f1_meter.update(f1.item(), images.size(0))

    # Calculate average metrics for the test set
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / test_total_pixels
    test_dice = test_dice_meter.avg
    test_iou = test_iou_meter.avg
    test_precision = test_precision_meter.avg
    test_recall = test_recall_meter.avg
    test_f1 = test_f1_meter.avg

    print(f'Test Results for {checkpoint_path}:')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Dice: {test_dice:.4f}, Test IoU: {test_iou:.4f}, '
          f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

    # Log the test metrics to TensorBoard
    writer.add_scalar('Dice Loss/test', test_loss, global_step=epoch)
    writer.add_scalar('Accuracy/test', test_acc, global_step=epoch)
    writer.add_scalar('Dice/test', test_dice, global_step=epoch)
    writer.add_scalar('IoU/test', test_iou, global_step=epoch)
    writer.add_scalar('Precision/test', test_precision, global_step=epoch)
    writer.add_scalar('Recall/test', test_recall, global_step=epoch)
    writer.add_scalar('F1/test', test_f1, global_step=epoch)

############################# TESTING FUNCTION (DIRECTLY IN MEMORY) ################################
def test_model_in_memory(model, device, test_loader, criterion, epoch):
    model.eval()

    test_loss = 0.0
    test_corrects = 0
    test_total_pixels = 0

    test_dice_meter = AverageMeter()
    test_iou_meter = AverageMeter()
    test_precision_meter = AverageMeter()
    test_recall_meter = AverageMeter()
    test_f1_meter = AverageMeter()

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = outputs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            test_corrects += (np.argmax(preds, axis=1) == masks_np).sum()
            test_total_pixels += masks_np.size

            # Calculate metrics
            dice_acc = dice_fn(outputs, masks)
            iou_acc = iou_fn(outputs, masks)

            preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)
            precision = precision_fn(preds_tensor, masks)
            recall = recall_fn(preds_tensor, masks)
            f1 = f1_fn(preds_tensor, masks)

            test_dice_meter.update(dice_acc.item(), images.size(0))
            test_iou_meter.update(iou_acc.item(), images.size(0))
            test_precision_meter.update(precision.item(), images.size(0))
            test_recall_meter.update(recall.item(), images.size(0))
            test_f1_meter.update(f1.item(), images.size(0))

    # Calculate average metrics for the test set
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / test_total_pixels
    test_dice = test_dice_meter.avg
    test_iou = test_iou_meter.avg
    test_precision = test_precision_meter.avg
    test_recall = test_recall_meter.avg
    test_f1 = test_f1_meter.avg

    print(f'[TEST in-memory] Epoch {epoch+1}: '
          f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Dice: {test_dice:.4f}, '
          f'IoU: {test_iou:.4f}, Precision: {test_precision:.4f}, '
          f'Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    # Log the test metrics to TensorBoard
    writer.add_scalar('Dice Loss/test', test_loss, global_step=epoch)
    writer.add_scalar('Accuracy/test', test_acc, global_step=epoch)
    writer.add_scalar('Dice/test', test_dice, global_step=epoch)
    writer.add_scalar('IoU/test', test_iou, global_step=epoch)
    writer.add_scalar('Precision/test', test_precision, global_step=epoch)
    writer.add_scalar('Recall/test', test_recall, global_step=epoch)
    writer.add_scalar('F1/test', test_f1, global_step=epoch)

############################# TESTING FUNCTION ################################
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

############################# TRAINING ################################

# Initialize variables for model checkpointing
best_val_loss = float('inf')
# Training loop
for epoch in range(EPOCHS):
    unet.train()
    running_loss = 0.0
    running_dice_meter.reset()
    running_iou_meter.reset()
    running_precision_meter.reset()
    running_recall_meter.reset()
    running_f1_meter.reset()
    running_corrects = 0
    total_pixels = 0

    for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Ensure the target masks contain only valid class indices
        if masks.max() >= NUM_CLASSES or masks.min() < 0:
            raise ValueError(f"Target mask contains invalid class indices: {masks.unique().tolist()}")

        optimizer.zero_grad()
        outputs = unet(images)
        
        # Ensure the output shape is [batch_size, num_classes, height, width]
        if outputs.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected output shape [batch_size, {NUM_CLASSES}, height, width], but got {outputs.shape}")

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        preds = outputs.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        running_corrects += (np.argmax(preds, axis=1) == masks_np).sum()
        total_pixels += masks_np.size
        
        # Calculate metrics
        dice_acc = dice_fn(outputs, masks)
        iou_acc = iou_fn(outputs, masks)
        
        preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)
        precision = precision_fn(preds_tensor, masks)
        recall = recall_fn(preds_tensor, masks)
        f1 = f1_fn(preds_tensor, masks)

        running_dice_meter.update(dice_acc.item(), images.size(0))
        running_iou_meter.update(iou_acc.item(), images.size(0))
        running_precision_meter.update(precision.item(), images.size(0))
        running_recall_meter.update(recall.item(), images.size(0))
        running_f1_meter.update(f1.item(), images.size(0))

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / total_pixels
    epoch_dice = running_dice_meter.avg
    epoch_iou = running_iou_meter.avg
    epoch_precision = running_precision_meter.avg
    epoch_recall = running_recall_meter.avg
    epoch_f1 = running_f1_meter.avg
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')

    # Step the learning rate scheduler
    scheduler.step()
    # scheduler.step(epoch + 1)

    # Log the current learning rate to TensorBoard
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('Learning Rate', current_lr, epoch)
    print(f"Current learning rate: {current_lr:.6f}")

    # Log the loss, accuracy, Dice, and IoU to TensorBoard
    writer.add_scalar('Dice Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    writer.add_scalar('Dice/train', epoch_dice, epoch)
    writer.add_scalar('IoU/train', epoch_iou, epoch)
    writer.add_scalar('Precision/train', epoch_precision, epoch)
    writer.add_scalar('Recall/train', epoch_recall, epoch)
    writer.add_scalar('F1/train', epoch_f1, epoch)

    ############################# VALIDATION ################################

    # Validation
    unet.eval()
    val_loss = 0.0
    val_corrects = 0
    val_dice_meter.reset()
    val_iou_meter.reset()
    val_precision_meter.reset()
    val_recall_meter.reset()
    val_f1_meter.reset()
    val_total_pixels = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = unet(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = outputs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            val_corrects += (np.argmax(preds, axis=1) == masks_np).sum()
            val_total_pixels += masks_np.size

            # Calculate metrics
            dice_acc = dice_fn(outputs, masks)
            iou_acc = iou_fn(outputs, masks)

            preds_tensor = torch.tensor(preds, dtype=torch.float32).to(device)
            precision = precision_fn(preds_tensor, masks)
            recall = recall_fn(preds_tensor, masks)
            f1 = f1_fn(preds_tensor, masks)

            val_dice_meter.update(dice_acc.item(), images.size(0))
            val_iou_meter.update(iou_acc.item(), images.size(0))
            val_precision_meter.update(precision.item(), images.size(0))
            val_recall_meter.update(recall.item(), images.size(0))
            val_f1_meter.update(f1.item(), images.size(0))

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects / val_total_pixels
    val_dice = val_dice_meter.avg
    val_iou = val_iou_meter.avg
    val_precision = val_precision_meter.avg
    val_recall = val_recall_meter.avg
    val_f1 = val_f1_meter.avg
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation Dice: {val_dice:.4f}, Validation IoU: {val_iou:.4f}', f'Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}')

    # Log the validation loss, accuracy, Dice, and IoU to TensorBoard
    writer.add_scalar('Dice Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Dice/val', val_dice, epoch)
    writer.add_scalar('IoU/val', val_iou, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1/val', val_f1, epoch)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1

        # Construct the best model path with epoch in the name
        best_model_path = os.path.join(models_dir, f'{HYPERPAR}_best_unet_model_at{best_epoch}epochs_{EPOCHS}epochs_{SUBSET}%subset_{config}.pth')

        # Remove any existing best model file to avoid clutter
        for file in os.listdir(models_dir):
            if file.startswith(f'{HYPERPAR}_best_unet_model_at') and file.endswith(f'{EPOCHS}epochs_{SUBSET}%subset_{config}.pth'):
                os.remove(os.path.join(models_dir, file))

        # Save the current best model
        torch.save(unet.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Save model at specific epochs
    # if (epoch + 1) in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    #     checkpoint_path = os.path.join(models_dir, f'unet_checkpoint_epoch{epoch + 1}_{config}.pth')
    #     torch.save(unet.state_dict(), checkpoint_path)
    #     print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

        # We no longer call test_model(checkpoint_path) here to avoid loading from disk.
        # If you want to test using the *current in-memory model*:
        # test_model_in_memory(unet, device, test_loader, criterion, epoch)
    
    # NEW: If you want to test on the test set every epoch (in-memory), do it here:
    # test_model_in_memory(unet, device, test_loader, criterion, epoch)

    if epoch == EPOCHS - 1 or epoch + 1 == best_epoch:
        test_model(unet, device, test_loader, criterion)

# Save the final U-Net model
num_images_str = NUM_IMAGES if NUM_IMAGES is not None else 'all'
final_model_path = os.path.join(models_dir, f'{HYPERPAR}_final_unet_model_{EPOCHS}epochs_{SUBSET}%subset_{config}.pth')
torch.save(unet.state_dict(), final_model_path)
print(f"Final U-Net model saved to {final_model_path}")

# Close the TensorBoard writer
writer.close()

############################# VISUALISATION #############################

# Define a colormap for the classes
class_colors = [
    (65, 155, 223),  # water
    (57, 125, 73),   # trees
    (136, 176, 83),  # grass
    (122, 135, 198), # flooded_vegetation
    (228, 150, 53),  # crops
    (223, 195, 90),  # shrub_and_scrub
    (196, 40, 27),   # built
    (165, 155, 143), # bare
    (179, 159, 225)  # snow_and_ice
]

class_names = [
    "Water",
    "Trees",
    "Grass",
    "Flooded Vegetation",
    "Crops",
    "Shrub & Scrub",
    "Built",
    "Bare",
    "Snow & ice"
]

def apply_colormap(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(class_colors):
        color_mask[mask == class_idx] = color
    return color_mask

# Load the Unet model
unet.load_state_dict(torch.load(final_model_path))
unet.eval()

# Plot some results
image_number = random.randint(0, len(test_dataset) - 1)
test_img, ground_truth = test_dataset[image_number]
raw_image, _ = raw_dataset[image_number]

test_img_input = test_img.unsqueeze(0).to(device)

# Perform prediction with probabilities
with torch.no_grad():
    outputs = unet(test_img_input).squeeze(0)  # shape: (C, H, W)
    probabilities = torch.softmax(outputs, dim=0).cpu().numpy()  # Get probabilities per class
    prediction = np.argmax(probabilities, axis=0)  # Predicted class per pixel
    predicted_prob = np.max(probabilities, axis=0) # Probability of chosen class per pixel
    print("Prediction:", prediction)

# Ensure ground_truth is 2D
ground_truth = ground_truth.squeeze(0)
ground_truth_np = ground_truth.numpy()

# Apply colormap to the ground truth and prediction
ground_truth_colored = apply_colormap(ground_truth_np)
prediction_colored = apply_colormap(prediction)

# Create correctness visualization
correct_mask = (prediction == ground_truth_np)
# Map correctness: correct predictions -> +p in [0,1], incorrect predictions -> -p in [-1,0]
correctness_map = np.where(correct_mask, predicted_prob, -predicted_prob)

# Apply contrast stretching to the original image
def contrast_stretch(image):
    p2, p98 = np.percentile(image, (2, 98))
    return np.clip((image - p2) / (p98 - p2), 0, 1)

test_img_rgb = test_img[:3].permute(1, 2, 0).cpu().numpy()
test_img_rgb = test_img_rgb[:, :, [2, 1, 0]]
test_img_rgb = contrast_stretch(test_img_rgb)

# Plot the images with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(test_img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(ground_truth_colored)
axes[1].set_title('Ground Truth')
axes[1].axis('off')

axes[2].imshow(prediction_colored)
axes[2].set_title('Prediction')
axes[2].axis('off')

# Display correctness map using a diverging colormap
im_corr = axes[3].imshow(correctness_map, cmap='RdYlGn', aspect='auto')
axes[3].set_title('Prediction Correctness')
axes[3].axis('off')

# Add class legend (already defined)
legend_patches = [mpatches.Patch(color=np.array(color)/255, label=class_name)
                  for color, class_name in zip(class_colors, class_names)]

# Instead of placing the legend inside a single subplot, 
# we create a combined legend outside the plotting area on the right.
fig.legend(handles=legend_patches, title='Classes', bbox_to_anchor=(1.05, 0.7), loc='center', borderaxespad=0.)

# Add a colorbar for correctness_map to show intensity
cbar = fig.colorbar(im_corr, ax=axes[3], fraction=0.046, pad=0.04)
cbar.set_label('Certainty', rotation=90)

# Create a small legend indicating correct/incorrect direction
# We'll use a small text explanation or two patches.
correct_patch = mpatches.Patch(color='green', label='Correctly classified pixels')
incorrect_patch = mpatches.Patch(color='red', label='Incorrectly classified pixels')

fig.legend(handles=[correct_patch, incorrect_patch],
           title='Correctness Scale',
           bbox_to_anchor=(1.05, 0.3), loc='center', borderaxespad=0.)

plt.tight_layout(w_pad=0.1)

# Save the image
output_path = os.path.join(output_dir, f"unet_results_{EPOCHS}epochs_{SUBSET}%subset_{config}.png")
plt.savefig(output_path, bbox_inches='tight')
print(f"Image saved to {output_path}")

############################# CALCULATE FINAL PERFORMANCE METRICS #############################

iou = calculate_iou(prediction, ground_truth.numpy())
print("Mean IoU =", iou)

# Calculate IoU for all test images and average
iou_values = []
for img, gt in val_dataset:
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = unet(img_input).squeeze(0).cpu().numpy()
        pred = np.argmax(pred, axis=0)  # Get the predicted class for each pixel
    iou_calc = calculate_iou(pred, gt.numpy())
    iou_values.append(iou_calc)

mean_iou = np.nanmean(iou_values)
print("Mean IoU is: ", mean_iou)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time in hours, minutes, and seconds
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total time taken to run script: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
