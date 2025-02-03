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
from PIL import Image
import seaborn as sns

# Record the start time
start_time = time.time()

############################# SETTING UP THE PARAMETERS #############################

SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 9
NUM_IMAGES = None
SUBSET = 100
SEED = 42

config = 'pretrain_S2'
# config = 'nopretrain_S2'

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
subset_file = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal_Biesbosch/subset_indices_per_split_S2.json"

# Load the subset indices
with open(subset_file, "r") as f:
    subset_indices = json.load(f)

# Choose the fraction of data to use (e.g., "10%", "50%", "100%")
fraction_key = f'{SUBSET}%'

train_indices = subset_indices["train"][fraction_key]
val_indices = subset_indices["val"][fraction_key]
test_indices = subset_indices["test"][fraction_key]

############################# LOADING THE DATA #############################

# Ensure the saved_images directory exists
output_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal_Biesbosch/saved_images_S2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure the models directory exists
models_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal_Biesbosch/models_S2"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load data for U-net training
train_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_images_train'
train_masks_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_masks_train'
val_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_images_val'
val_masks_dir =  '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_masks_val'
test_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_images_test'
test_masks_dir =  '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_S2/Biesbosch_masks_test'

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

        mask = np.array(Image.open(mask_path))

        # Clip pixel values to valid range [0, 10000]
        image = np.clip(image, 0, 10000)

        # Normalize the image
        image = image.astype(np.float32) / 10000.0

        if self.transform:
            image = self.transform(image)

        # # Print unique values in the mask
        # unique_values = np.unique(mask)
        # print(f"[Image {idx}] Unique values in mask: {unique_values}")

        mask = mask.astype(np.int16)

        # Exclude "Negative" class (set to -1 to ignore during loss computation)
        mask[mask == 0] = -1  # Replace 0 with -1 (PyTorch ignores -1 in loss functions)

        # # Print unique values in the mask
        # unique_values = np.unique(mask)
        # print(f"[Image {idx}] Unique values in mask: {unique_values}")

        # Ensure mask is not normalized and is of type long
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
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

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, ignore_index=-1):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index

#     def forward(self, outputs, targets):
#         # Apply softmax to outputs for multi-class segmentation
#         outputs = F.softmax(outputs, dim=1)

#         # Mask out ignore_index
#         valid_mask = (targets != self.ignore_index)
#         outputs = outputs * valid_mask.unsqueeze(1)
#         targets = targets * valid_mask

#         # Flatten the tensors
#         outputs = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
#         targets = targets.contiguous().view(targets.size(0), 1, -1)

#         # One-hot encode the targets
#         targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets, 1)

#         # Compute the Dice coefficient per class
#         intersection = (outputs * targets_one_hot).sum(dim=2)
#         union = outputs.sum(dim=2) + targets_one_hot.sum(dim=2)

#         dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

#         # Return the average Dice Loss (1 - Dice coefficient)
#         return 1 - dice_score.mean()

# class WeightedDiceLoss(nn.Module):
#     def __init__(self, class_order, num_classes, smooth=1e-6, ignore_index=-1):
#         """
#         Weighted Dice Loss with class-specific weights.

#         Args:
#             class_order (list): A list defining the order of importance for class indices.
#             num_classes (int): Total number of classes.
#             smooth (float): Smoothing factor to avoid division by zero.
#             ignore_index (int): Index to ignore in the loss calculation.
#         """
#         super(WeightedDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.num_classes = num_classes

#         # Assign weights in ascending order of importance
#         weights = torch.tensor([i + 1 for i in range(len(class_order))], dtype=torch.float32)

#         # Initialize a weights tensor for all classes
#         self.class_weights = torch.ones(num_classes, dtype=torch.float32)

#         # Assign the weights to the specified classes
#         for idx, cls in enumerate(class_order):
#             self.class_weights[cls] = weights[idx]

#     def forward(self, outputs, targets):
#         # Move class_weights to the same device as outputs
#         self.class_weights = self.class_weights.to(outputs.device)

#         # Apply softmax to outputs for multi-class segmentation
#         outputs = F.softmax(outputs, dim=1)

#         # Mask out ignore_index
#         valid_mask = (targets != self.ignore_index)
#         outputs = outputs * valid_mask.unsqueeze(1)
#         targets = targets * valid_mask

#         # Flatten the tensors
#         outputs = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
#         targets = targets.contiguous().view(targets.size(0), 1, -1)

#         # One-hot encode the targets
#         targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets, 1)

#         # Compute the Dice coefficient per class
#         intersection = (outputs * targets_one_hot).sum(dim=2)
#         union = outputs.sum(dim=2) + targets_one_hot.sum(dim=2)
#         dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

#         # Apply class weights
#         weighted_dice = (self.class_weights * (1 - dice_score)).mean()
#         return weighted_dice



class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, ignore_index=-1):
        """
        Weighted Dice Loss with class-specific weights.

        Args:
            num_classes (int): Total number of classes.
            smooth (float): Smoothing factor to avoid division by zero.
            ignore_index (int): Index to ignore in the loss calculation.
        """
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        # Define class weights
        self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        # Reduce weights for specific classes (Forest, Grass & Farmland, Water)
        class_indices_to_reduce = [4, 8]  # Indices of Grass & Farmland, Water
        for idx in class_indices_to_reduce:
            self.class_weights[idx] *= 0.2  # Reduce weight by a factor of 10

    def forward(self, outputs, targets):
        # Move class_weights to the same device as outputs
        self.class_weights = self.class_weights.to(outputs.device)

        # Apply softmax to outputs for multi-class segmentation
        outputs = F.softmax(outputs, dim=1)

        # Mask out ignore_index
        valid_mask = (targets != self.ignore_index)
        outputs = outputs * valid_mask.unsqueeze(1)
        targets = targets * valid_mask

        # Flatten the tensors
        outputs = outputs.contiguous().view(outputs.size(0), outputs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), 1, -1)

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets, 1)

        # Compute the Dice coefficient per class
        intersection = (outputs * targets_one_hot).sum(dim=2)
        union = outputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        # Apply class weights
        weighted_dice = (self.class_weights * (1 - dice_score)).mean()
        return weighted_dice
    
# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = Unet(num_classes=NUM_CLASSES).to(device)  # Ensure the model is initialized with the correct number of classes
# criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
# class_order = [4, 8, 3, 6, 1, 2, 7]
# criterion = WeightedDiceLoss(class_order=class_order, num_classes=NUM_CLASSES).to(device)
criterion = WeightedDiceLoss(num_classes=NUM_CLASSES).to(device)
# criterion = DiceLoss()  
optimizer = optim.AdamW(unet.parameters(), lr=0.0001, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize TensorBoard writer with date and time in the run name
log_dir = f'/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal_Biesbosch/runs_S2/train_unet_{config}_{current_time}'
writer = SummaryWriter(log_dir=log_dir)

# Load pre-trained encoder weights
if config == 'pretrain_S2':
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
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth or prediction, do not include in IoU calculation
        else:
            iou_list.append(intersection / union)
    return np.nanmean(iou_list)

############################# TESTING FUNCTION ################################

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

            # Forward pass
            outputs = unet(images)
            outputs[:, 0, :, :] = -float('inf')
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)  # Get predicted class per pixel
            masks_np = masks.cpu().numpy()
            test_corrects += (preds.cpu().numpy() == masks_np).sum()
            test_total_pixels += masks_np.size

            # Create valid mask to exclude ignored indices (-1)
            valid_mask = masks != -1

            # Apply valid mask to predictions and ground truth
            valid_preds = preds[valid_mask]
            valid_masks = masks[valid_mask]

            # Calculate metrics only on valid pixels
            dice_acc = dice_fn(valid_preds, valid_masks)
            iou_acc = iou_fn(valid_preds, valid_masks)

            # Convert predictions to tensor for additional metrics
            preds_tensor = outputs.softmax(dim=1).detach()
            valid_preds_tensor = preds_tensor[:, :, valid_mask].reshape(-1, preds_tensor.shape[1])

            precision = precision_fn(valid_preds_tensor, valid_masks)
            recall = recall_fn(valid_preds_tensor, valid_masks)
            f1 = f1_fn(valid_preds_tensor, valid_masks)

            test_dice_meter.update(dice_acc.item(), valid_preds.size(0))
            test_iou_meter.update(iou_acc.item(), valid_preds.size(0))
            test_precision_meter.update(precision.item(), valid_preds.size(0))
            test_recall_meter.update(recall.item(), valid_preds.size(0))
            test_f1_meter.update(f1.item(), valid_preds.size(0))

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
            outputs[:, 0, :, :] = -float('inf')  # Ignore the "Negative" class if applicable
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)  # Get class predictions
            valid_mask = masks != -1  # Create a mask for valid pixels

            # Mask predictions and ground truth
            valid_preds = preds[valid_mask]
            valid_masks = masks[valid_mask]

            if valid_preds.numel() == 0:
                # Skip metric calculations if no valid pixels
                continue

            # Calculate test accuracy
            test_corrects += (valid_preds == valid_masks).sum().item()
            test_total_pixels += valid_mask.sum().item()

            # Calculate metrics
            dice_acc = dice_fn(valid_preds, valid_masks)
            iou_acc = iou_fn(valid_preds, valid_masks)

            precision = precision_fn(valid_preds, valid_masks)
            recall = recall_fn(valid_preds, valid_masks)
            f1 = f1_fn(valid_preds, valid_masks)

            # Update metric trackers
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
    
    # Initialize metrics
    test_loss = 0.0
    test_corrects = 0
    test_total_pixels = 0
    
    # Initialize AverageMeter objects
    test_dice_meter = AverageMeter()
    test_iou_meter = AverageMeter()
    test_precision_meter = AverageMeter()
    test_recall_meter = AverageMeter()
    test_f1_meter = AverageMeter()

    # Initialize confusion matrix and per-class counters
    conf_matrix_test = np.zeros((NUM_CLASSES, NUM_CLASSES))
    correct_per_class_test = [0] * NUM_CLASSES
    total_per_class_test = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            outputs[:, 0, :, :] = -float('inf')  # Ignore the "Negative" class if applicable
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)

            # Get predictions and create a valid mask
            preds = torch.argmax(outputs, dim=1)  # Get predicted class per pixel
            valid_mask = masks != -1  # Ignore invalid pixels (-1)

            # Filter predictions and ground truth using the valid mask
            valid_preds = preds[valid_mask]
            valid_masks = masks[valid_mask]

            if valid_preds.numel() == 0:
                # Skip metric calculations if no valid pixels
                continue

            # Update per-class metrics and confusion matrix
            update_metrics(
                valid_preds.cpu().numpy(),
                valid_masks.cpu().numpy(),
                correct_per_class_test,
                total_per_class_test,
                conf_matrix_test
            )

            # Calculate overall metrics
            dice_acc = dice_fn(valid_preds, valid_masks)
            iou_acc = iou_fn(valid_preds, valid_masks)
            precision = precision_fn(valid_preds, valid_masks)
            recall = recall_fn(valid_preds, valid_masks)
            f1 = f1_fn(valid_preds, valid_masks)

            test_dice_meter.update(dice_acc.item(), images.size(0))
            test_iou_meter.update(iou_acc.item(), images.size(0))
            test_precision_meter.update(precision.item(), images.size(0))
            test_recall_meter.update(recall.item(), images.size(0))
            test_f1_meter.update(f1.item(), images.size(0))

            # Calculate accuracy for valid pixels
            test_corrects += (valid_preds == valid_masks).sum().item()
            test_total_pixels += valid_mask.sum().item()

    # Calculate averages for the test set
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / test_total_pixels
    test_dice = test_dice_meter.avg
    test_iou = test_iou_meter.avg
    test_precision = test_precision_meter.avg
    test_recall = test_recall_meter.avg
    test_f1 = test_f1_meter.avg

    print(f'Test Results:')
    print(f'Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, IoU: {test_iou:.4f}, '
          f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    # Calculate and print per-class and weighted accuracy
    test_per_class_accuracy = calculate_per_class_accuracy(correct_per_class_test, total_per_class_test)
    test_weighted_accuracy = calculate_weighted_accuracy(correct_per_class_test, total_per_class_test)

    print(f'Weighted Accuracy (Test): {test_weighted_accuracy:.4f}')
    print('Per-Class Accuracy (Test):')
    print_per_class_accuracy(test_per_class_accuracy, class_names)

    # Print and save the confusion matrix
    print('Test Confusion Matrix:')
    print_confusion_matrix(conf_matrix_test, class_names)
    plot_confusion_matrix(conf_matrix_test, class_names, 'Test Confusion Matrix',
                          os.path.join(output_dir, f'test_conf_matrix_{config}.png'))

############################# TRAINING ################################

# Define class names
class_names = [
    "Negative",
    "Built",
    "Flooded Soil",
    "Forest",
    "Grass & Farmland",
    "Invalid Pixels",
    "Reed & Rough",
    "Shrubs",
    "Water"
]
# Initialize counters for each class
correct_per_class_train = [0] * NUM_CLASSES
total_per_class_train = [0] * NUM_CLASSES
correct_per_class_val = [0] * NUM_CLASSES
total_per_class_val = [0] * NUM_CLASSES
correct_per_class_test = [0] * NUM_CLASSES
total_per_class_test = [0] * NUM_CLASSES

# Initialize confusion matrices
conf_matrix_train = np.zeros((NUM_CLASSES, NUM_CLASSES))
conf_matrix_val = np.zeros((NUM_CLASSES, NUM_CLASSES))
conf_matrix_test = np.zeros((NUM_CLASSES, NUM_CLASSES))

# Function to update counters and confusion matrix
def update_metrics(predictions, labels, correct_per_class, total_per_class, conf_matrix):
    for pred, label in zip(predictions, labels):
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1
        conf_matrix[label, pred] += 1

# Function to calculate per-class accuracy
def calculate_per_class_accuracy(correct_per_class, total_per_class):
    return [correct / total if total > 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]

# Function to print per-class accuracy with class names
def print_per_class_accuracy(per_class_accuracy, class_names):
    for class_name, accuracy in zip(class_names, per_class_accuracy):
        print(f'{class_name}: {accuracy:.4f}')

def plot_confusion_matrix(conf_matrix, class_names, title, output_path):
    plt.figure(figsize=(12, 10))  # Increase the figure size
    ax = sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})  # Reduce font size of annotations
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted class of pixel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground truth class of pixel', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0, fontsize=10)  # Set y-axis labels font size
    plt.tight_layout()  # Adjust layout to fit everything
    plt.savefig(output_path)
    plt.close()

# Function to print confusion matrix
def print_confusion_matrix(conf_matrix, class_names):
    print("Confusion Matrix:")
    print(" " * 10 + " ".join(f"{name[:5]:>5}" for name in class_names))
    for i, row in enumerate(conf_matrix):
        print(f"{class_names[i]:<10} " + " ".join(f"{int(val):>5}" for val in row))

# Function to calculate weighted accuracy
def calculate_weighted_accuracy(correct_per_class, total_per_class):
    total_correct = sum(correct_per_class)
    total_pixels = sum(total_per_class)
    return total_correct / total_pixels if total_pixels > 0 else 0

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
        if masks.max() >= NUM_CLASSES or masks.min() < -1:
            raise ValueError(f"Target mask contains invalid class indices: {masks.unique().tolist()}")

        optimizer.zero_grad()
        outputs = unet(images)

        # Ensure the output shape is [batch_size, num_classes, height, width]
        if outputs.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected output shape [batch_size, {NUM_CLASSES}, height, width], but got {outputs.shape}")

        # Calculate the loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Calculate predictions
        preds = torch.argmax(outputs, dim=1)
        valid_mask = masks != -1  # Create a mask for valid pixels

        # Mask predictions and ground truth
        valid_preds = preds[valid_mask]
        valid_masks = masks[valid_mask]

        if valid_preds.numel() == 0:
            # Skip metric calculations if no valid pixels
            continue

        # Calculate accuracy
        running_corrects += (valid_preds == valid_masks).sum().item()
        total_pixels += valid_mask.sum().item()

        # Calculate metrics
        dice_acc = dice_fn(valid_preds, valid_masks)
        iou_acc = iou_fn(valid_preds, valid_masks)
        precision = precision_fn(valid_preds, valid_masks)
        recall = recall_fn(valid_preds, valid_masks)
        f1 = f1_fn(valid_preds, valid_masks)

        # Update metric trackers
        running_dice_meter.update(dice_acc.item(), images.size(0))
        running_iou_meter.update(iou_acc.item(), images.size(0))
        running_precision_meter.update(precision.item(), images.size(0))
        running_recall_meter.update(recall.item(), images.size(0))
        running_f1_meter.update(f1.item(), images.size(0))

        # Update train metrics
        update_metrics(valid_preds.cpu().numpy(), valid_masks.cpu().numpy(), correct_per_class_train, total_per_class_train, conf_matrix_train)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / total_pixels
    epoch_dice = running_dice_meter.avg
    epoch_iou = running_iou_meter.avg
    epoch_precision = running_precision_meter.avg
    epoch_recall = running_recall_meter.avg
    epoch_f1 = running_f1_meter.avg
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')

    train_per_class_accuracy = calculate_per_class_accuracy(correct_per_class_train, total_per_class_train)
    print('Per-Class Accuracy (Train):')
    print_per_class_accuracy(train_per_class_accuracy, class_names)

    train_weighted_accuracy = calculate_weighted_accuracy(correct_per_class_train, total_per_class_train)
    print(f'Weighted Accuracy (Train): {train_weighted_accuracy:.4f}')

    # # Step the learning rate scheduler
    # scheduler.step()
    # # scheduler.step(epoch + 1)

    # # Log the current learning rate to TensorBoard
    # current_lr = scheduler.get_last_lr()[0]
    # writer.add_scalar('Learning Rate', current_lr, epoch)
    # print(f"Current learning rate: {current_lr:.6f}")

    # Log the loss, accuracy, Dice, and IoU to TensorBoard
    writer.add_scalar('Dice Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    writer.add_scalar('Dice/train', epoch_dice, epoch)
    writer.add_scalar('IoU/train', epoch_iou, epoch)
    writer.add_scalar('Precision/train', epoch_precision, epoch)
    writer.add_scalar('Recall/train', epoch_recall, epoch)
    writer.add_scalar('F1/train', epoch_f1, epoch)
    writer.add_scalar('Weighted Accuracy/train', train_weighted_accuracy, epoch)

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
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = unet(images)

            # Apply a large negative value to the background class if necessary
            outputs[:, 0, :, :] = -float('inf')

            # Compute the loss
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)
            valid_mask = masks != -1  # Create a mask for valid pixels

            # Mask predictions and ground truth
            valid_preds = preds[valid_mask]
            valid_masks = masks[valid_mask]

            if valid_preds.numel() == 0:
                # Skip metric calculations if no valid pixels
                continue

            # Calculate accuracy
            val_corrects += (valid_preds == valid_masks).sum().item()
            val_total_pixels += valid_mask.sum().item()

            # Calculate metrics
            dice_acc = dice_fn(valid_preds, valid_masks)
            iou_acc = iou_fn(valid_preds, valid_masks)
            precision = precision_fn(valid_preds, valid_masks)
            recall = recall_fn(valid_preds, valid_masks)
            f1 = f1_fn(valid_preds, valid_masks)

            # Update metric trackers
            val_dice_meter.update(dice_acc.item(), images.size(0))
            val_iou_meter.update(iou_acc.item(), images.size(0))
            val_precision_meter.update(precision.item(), images.size(0))
            val_recall_meter.update(recall.item(), images.size(0))
            val_f1_meter.update(f1.item(), images.size(0))

            # Update validation metrics
            update_metrics(valid_preds.cpu().numpy(), valid_masks.cpu().numpy(), correct_per_class_val, total_per_class_val, conf_matrix_val)

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects / val_total_pixels
    val_dice = val_dice_meter.avg
    val_iou = val_iou_meter.avg
    val_precision = val_precision_meter.avg
    val_recall = val_recall_meter.avg
    val_f1 = val_f1_meter.avg
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation Dice: {val_dice:.4f}, Validation IoU: {val_iou:.4f}', f'Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}')

    val_per_class_accuracy = calculate_per_class_accuracy(correct_per_class_val, total_per_class_val)
    print('Validation Per-Class Accuracy:')
    print_per_class_accuracy(val_per_class_accuracy, class_names)

    val_weighted_accuracy = calculate_weighted_accuracy(correct_per_class_val, total_per_class_val)
    print(f'Weighted Accuracy (Validation): {val_weighted_accuracy:.4f}')

    # Log the validation loss, accuracy, Dice, and IoU to TensorBoard
    writer.add_scalar('Dice Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Dice/val', val_dice, epoch)
    writer.add_scalar('IoU/val', val_iou, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('Weighted Accuracy/val', val_weighted_accuracy, epoch)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model_path = os.path.join(models_dir, f'best_unet_model_{config}.pth')
        torch.save(unet.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    
        # Print confusion matrix and per-class accuracy for the best epoch
        print('Best Epoch Confusion Matrix (Validation):')
        print_confusion_matrix(conf_matrix_val, class_names)
        print('Best Epoch Per-Class Accuracy (Validation):')
        print_per_class_accuracy(val_per_class_accuracy, class_names)

        # Save confusion matrix as PNG
        plot_confusion_matrix(conf_matrix_val, class_names, 'Best Epoch Confusion Matrix (Validation)', os.path.join(output_dir, f'best_epoch_conf_matrix_{config}.png'))

    # # Save model at specific epochs
    # # if (epoch + 1) in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    # if (epoch + 1) in [1, 10, 20, 30, 40, 50]:
    #     checkpoint_path = os.path.join(models_dir, f'unet_checkpoint_epoch{epoch + 1}_{config}.pth')
    #     torch.save(unet.state_dict(), checkpoint_path)
    #     print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

        # test_model(checkpoint_path)
    
    # test_model_in_memory(unet, device, test_loader, criterion, epoch)

    if epoch == EPOCHS - 1:
        # Print confusion matrix and per-class accuracy for the last epoch
        print('Last Epoch Confusion Matrix (Validation):')
        print_confusion_matrix(conf_matrix_val, class_names)
        print('Last Epoch Per-Class Accuracy (Validation)):')
        print_per_class_accuracy(val_per_class_accuracy, class_names)

        # Save confusion matrix as PNG
        plot_confusion_matrix(conf_matrix_val, class_names, 'Last Epoch Confusion Matrix (Validation)', os.path.join(output_dir, f'last_epoch_conf_matrix_{config}.png'))

        test_model(unet, device, test_loader, criterion)

# Save the final U-Net model
num_images_str = NUM_IMAGES if NUM_IMAGES is not None else 'all'
final_model_path = os.path.join(models_dir, f'unet_model_weights_{EPOCHS}epochs_{SUBSET}%subset_{config}.pth')
torch.save(unet.state_dict(), final_model_path)
print(f"Final U-Net model saved to {final_model_path}")

# Close the TensorBoard writer
writer.close()

############################# VISUALISATION #############################

# Define a colormap for the classes
class_colors = [ 
    (0, 0, 0),       # Negative
    (255, 0, 0),     # Built
    (110, 93, 3),     # Flooded Soil
    (20, 102, 0),    # Forest
    (83, 251, 84),   # Grass & Farmland
    (255, 0, 204),   # Invalid Pixels 
    (224, 255, 6),    # Reed & Rough
    (255, 214, 0),   # Shrubs
    (0, 195, 206)   # Water
]

class_names = [
    "Negative",
    "Built",
    "Flooded Soil",
    "Forest",
    "Grass & Farmland",
    "Invalid Pixels",
    "Reed & Rough",
    "Shrubs",
    "Water"
]

def apply_colormap(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(class_colors):
        color_mask[mask == class_idx] = color
    return color_mask

############################# PLOTTING RESULTS #############################

# Load the Unet model
unet = Unet(num_classes=NUM_CLASSES).to(device)
unet.load_state_dict(torch.load(final_model_path))
unet.eval()

# # load unet model from saved .pth file
# unet = Unet(num_classes=NUM_CLASSES).to(device)
# unet.load_state_dict(torch.load('/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/models/unet_model_weights_10epochs_100%subset_nopretrain.pth'))
# unet.eval()

# Visualize and save results for multiple test images
num_images_to_visualize = 10  # Number of images to visualize
visualized_images = 0  # Counter for successfully visualized images

while visualized_images < num_images_to_visualize:
    image_number = random.randint(0, len(test_dataset) - 1)
    test_img, ground_truth = test_dataset[image_number]
    raw_image, _ = raw_dataset[image_number]

    # Check if more than 20% of ground truth pixels are non-negative
    non_negative_pixels = np.count_nonzero(ground_truth.numpy() > 0)
    total_pixels = ground_truth.numel()
    non_negative_percentage = (non_negative_pixels / total_pixels) * 100

    if non_negative_percentage < 20:
        # Skip this image if less than 20% of the pixels are non-negative
        continue

    test_img_input = test_img.unsqueeze(0).to(device)

    # Perform prediction with probabilities
    with torch.no_grad():
        outputs = unet(test_img_input).squeeze(0)  # shape: (C, H, W)
        outputs[0, :, :] = -float('inf')
        probabilities = torch.softmax(outputs, dim=0).cpu().numpy()  # Get probabilities per class
        prediction = np.argmax(outputs.detach().cpu().numpy(), axis=0) # Predicted class per pixel
        predicted_prob = np.max(probabilities, axis=0)  # Probability of chosen class per pixel

    # Print the probability distribution for the first 5 pixels
    # print(f"Probabilities for the first 5 pixels for {i}: {probabilities[:, :5, :5]}")

    # Ensure ground_truth is 2D
    ground_truth = ground_truth.squeeze(0)

    # Apply colormap to the ground truth and prediction
    ground_truth_colored = apply_colormap(ground_truth.numpy())
    prediction_colored = apply_colormap(prediction)

    # Create correctness visualization
    correct_mask = (prediction == ground_truth.numpy())
    correctness_map = np.where(correct_mask, predicted_prob, -predicted_prob)

    # Normalize raw image for visualization
    raw_test_image = raw_image.astype(np.float32)[:, :, :3]
    raw_test_image = (raw_test_image - raw_test_image.min()) / (raw_test_image.max() - raw_test_image.min())

    # Calculate the average probability for each class
    average_probabilities = probabilities.mean(axis=(1, 2))  # Average over the height and width

    # Apply contrast stretching to the original image
    def contrast_stretch(image):
        p2, p98 = np.percentile(image, (2, 98))
        return np.clip((image - p2) / (p98 - p2), 0, 1)

    test_img_rgb = test_img[:3].permute(1, 2, 0).cpu().numpy()
    test_img_rgb = test_img_rgb[:, :, [2, 1, 0]]
    test_img_rgb = contrast_stretch(test_img_rgb)

    # Create individual figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axes[0].imshow(raw_test_image, vmin=0, vmax=1)
    # axes[0].set_title(f'Raw Image ({test_image_files[image_number]})')
    # axes[0].axis('off')

    axes[0].imshow(test_img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ground_truth_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(prediction_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # im_corr = axes[3].imshow(correctness_map, cmap='RdYlGn', aspect='auto')
    # axes[3].set_title('Prediction Correctness')
    # axes[3].axis('off')

    # # Add legends for the figure
    # legend_patches = [mpatches.Patch(color=np.array(color) / 255, label=class_name) for color, class_name in zip(class_colors, class_names)]
    # fig.legend(handles=legend_patches, bbox_to_anchor=(1.05, 0.7), loc='center', borderaxespad=0., title='Classes')

    # Add legends for the figure with average probabilities
    legend_patches = []
    for class_idx, (color, class_name) in enumerate(zip(class_colors, class_names)):
        avg_prob = average_probabilities[class_idx]
        label = f"{class_name} (Avg Prob: {avg_prob:.2%})"
        legend_patches.append(mpatches.Patch(color=np.array(color) / 255, label=label))

    fig.legend(handles=legend_patches, bbox_to_anchor=(1.05, 0.5), loc='center', borderaxespad=0., title='Classes')

    #####

    # cbar = fig.colorbar(im_corr, ax=axes[3], fraction=0.046, pad=0.04)
    # cbar.set_label('Certainty', rotation=90)

    # correct_patch = mpatches.Patch(color='green', label='Correctly classified pixels')
    # incorrect_patch = mpatches.Patch(color='red', label='Incorrectly classified pixels')
    # fig.legend(handles=[correct_patch, incorrect_patch],
    #            title='Correctness Scale',
    #            bbox_to_anchor=(1.05, 0.3), loc='center', borderaxespad=0.)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 1]) 

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 1]) 

    # Save individual image
    output_path = os.path.join(output_dir, f"unet_results_image_{visualized_images + 1}_{EPOCHS}epochs_{SUBSET}%subset_{config}.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Image {visualized_images + 1} saved to {output_path}")

    plt.close(fig)

    visualized_images += 1  # Increment the visualized images counter

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
    iou = calculate_iou(pred, gt.numpy())
    iou_values.append(iou)

mean_iou = np.nanmean(iou_values)
print("Mean IoU is: ", mean_iou)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time in hours, minutes, and seconds
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total time taken to run script: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
