# https://youtu.be/hTpq9lzAb8M
"""
@author: Sreenivas Bhattiprolu

Train an autoencoder for a given type of images. e.g. EM images of cells.
Use the encoder part of the trained autoencoder as the encoder for a U-net.
Use pre-trained weights from autoencoder as starting weights for encoder in the Unet. 
Train the Unet.

Training with initial encoder pre-trained weights would dramatically speed up 
the training process of U-net. 

"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from models_015dropout import Autoencoder, Encoder, Unet
from torch.utils.tensorboard import SummaryWriter
import time
import rasterio
from datetime import datetime
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms.functional import equalize
import kornia.losses as losses
from kornia.metrics import ssim
import torch.nn.functional as F
import random
import json
from torch.utils.data import Subset

# Record the start time
start_time = time.time()

############################# SETTING UP THE PARAMETERS #############################

SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
NUM_IMAGES = None
SEED = 42
SUBSET = 100

HUBER = 0.5
SSIM = 0.4
EDGE = 0.1

RUNTYPE = f'{HUBER}_{SSIM}_{EDGE}_00001_015dropout_AdamW'.replace('.', '')

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

# Choose the fraction of data to use 
fraction_key = f'{SUBSET}%'  

train_indices = subset_indices["train"][fraction_key]
val_indices = subset_indices["val"][fraction_key]
test_indices = subset_indices["test"][fraction_key]

############################# LOADING THE DATA #############################

# Ensure the saved_images directory exists
output_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/saved_images_reconstructed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure the models directory exists
models_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load data for U-net training
train_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_train_S2'
val_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val_S2'
test_images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_test_S2'

# Print (image) size of one of the files in train_images_dir
img_path = os.path.join(train_images_dir, os.listdir(train_images_dir)[0])
with rasterio.open(img_path) as src:
    image = src.read()
print(f"Image size: {image.shape}")

############################# DEFINE HELPER AND DATA LOADING FUNCTIONS #############################

def save_combined_pixel_distribution(image, prefix='', suffix=''):
    """
    Create a combined histogram plot for all bands of the input image with descriptive
    channel names in the legend.

    Args:
        image (numpy.ndarray): Input image in HWC (Height x Width x Channels) format.
        prefix (str): Optional prefix for the output file name.
        suffix (str): Optional suffix for the output file name.
    """

    # Ensure the output directory exists
    output_dir = '/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/pixeldistribution'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the min and max pixel values dynamically
    pixel_min = image.min()
    pixel_max = image.max()

    # Number of bands (assuming HWC format)
    bands = image.shape[2]

    # Band names corresponding to indices
    band_names = {
        1: "B2 (Blue)",        # Band 1
        2: "B3 (Green)",       # Band 2
        3: "B4 (Red)",         # Band 3
        4: "B5 (Red Edge 1)",  # Band 4
        5: "B6 (Red Edge 2)",  # Band 5
        6: "B7 (Red Edge 3)",  # Band 6
        7: "B8 (NIR)",         # Band 7
        8: "B11 (SWIR 1)",     # Band 8
        9: "B12 (SWIR 2)"      # Band 9
    }

    # Colors for the selected bands
    colors = ['blue', 'green', 'red', '#9b59b6', '#8e44ad', '#6c3483', '#1abc9c', '#e67e22', '#d35400']

    # Plot histograms for all channels in one figure
    plt.figure(figsize=(10, 5))

    for i in range(bands):  # Ensure we only process existing bands
        band_label = band_names.get(i + 1, f"Band {i + 1}")  # Use i + 1 to map to band names
        plt.hist(image[:, :, i].ravel(), bins=100, range=(pixel_min, pixel_max), alpha=0.6, color=colors[i % len(colors)],
                 label=band_label)

    plt.title('Histogram After Equalization (Image)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    # Construct the file name with optional prefix and suffix
    file_name = f"{prefix}combined_pixeldistribution{suffix}.png"
    save_path = os.path.join(output_dir, file_name)

    # Save the combined histogram
    plt.savefig(save_path)
    plt.close()
    # print(f"Combined histogram saved to {save_path}")



def histogram_equalization_16bit(image):
    """
    Perform histogram equalization on a multi-channel 16-bit image.

    Parameters:
    - image: numpy array of shape (H, W, C) with pixel values in [0, 10000].

    Returns:
    - equalized_image: numpy array of same shape as image, with histogram equalized.
    """
    equalized_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):  # Process each channel independently
        channel = image[:, :, c]
        
        # Compute the histogram and cumulative distribution function (CDF)
        hist, bins = np.histogram(channel.flatten(), bins=10001, range=(0, 10000))
        cdf = hist.cumsum()
        cdf_normalized = cdf * (10000.0 / cdf[-1])  # Normalize CDF to range [0, 10000]
        
        # Use linear interpolation of the CDF to find new pixel values
        equalized_image[:, :, c] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)

    return equalized_image.astype(np.float32)  # Keep as float32 for further processing

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None, raw=False):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = image_files
        self.raw = raw

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Use rasterio to read the TIFF file
        with rasterio.open(img_path) as src:
            image = src.read()  # Read all bands
            image = np.transpose(image, (1, 2, 0))  # Transpose to HWC format

        if self.raw:
            print(f"[Image {idx}] Raw: min={image.min()}, max={image.max()}")
            return image

        # # For the first 5 images, save the pixel distribution
        # if idx < 10:
        #     save_combined_pixel_distribution(image, prefix=f"image_{idx}_", suffix="_before normalization")

        # print pixel range of raw image before transform per band
        # for band in range(image.shape[2]):
            # print(f"[Image {idx}] before normalization Band {band}: min={image[:, :, band].min()}, max={image[:, :, band].max()}")

        # Clip pixel values to valid range [0, 10000]
        image = np.clip(image, 0, 10000)

        # Perform histogram equalization
        image = histogram_equalization_16bit(image)

        # Normalize to [0, 1] for model input
        image = image / 10000.0

        # # For the first 5 images, save the pixel distribution
        # if idx < 10:
        #     save_combined_pixel_distribution(image, prefix=f"image_{idx}_", suffix="_after normalization")

        # print pixel range of raw image before transform per band
        # for band in range(image.shape[2]):
            # print(f"[Image {idx}] before transformation Band {band}: min={image[:, :, band].min()}, max={image[:, :, band].max()}")

        if self.transform:
            image = self.transform(image)

        # # print pixel range of image after transform per band
        # for band in range(image.shape[0]):
        #     print(f"[Image {idx}] after transformation Band {band}: min={image[band].min()}, max={image[band].max()}")

        return image

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Get the list of image files
train_image_files = sorted(os.listdir(train_images_dir))[:NUM_IMAGES]
val_image_files = sorted(os.listdir(val_images_dir))[:NUM_IMAGES]
test_image_files = sorted(os.listdir(test_images_dir))[:NUM_IMAGES]

print(f"Number of training images for autoencoder: {len(train_image_files)}")
print(f"Number of validation images for autoencoder: {len(val_image_files)}")
print(f"Number of test images for autoencoder: {len(test_image_files)}")

train_dataset = ImageDataset(train_images_dir, train_image_files, transform=transform)
val_dataset = ImageDataset(val_images_dir, val_image_files, transform=transform)
test_dataset = ImageDataset(test_images_dir, test_image_files, transform=transform)
raw_test_dataset = ImageDataset(image_dir=test_images_dir, image_files=test_image_files, transform=None, raw=True)

# Apply subsets using the loaded indices
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)
test_dataset = Subset(test_dataset, test_indices)
raw_test_dataset = Subset(raw_test_dataset, test_indices)

# Print the number of images in each subset
print(f"Number of training images in subset: {len(train_dataset)}")
print(f"Number of validation images in subset: {len(val_dataset)}")
print(f"Number of test images in subset: {len(test_dataset)}")

# Create DataLoaders for subsets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

############################# DEFINE TRAINING LOSS AND FUNCTIONS #############################

# Define MultiLoss
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # Compute gradients (edges) for predictions
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Compute gradients (edges) for targets
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Compute L1 loss between gradients
        loss_x = self.l1_loss(pred_dx, target_dx)
        loss_y = self.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y

class MultiLoss(nn.Module):
    def __init__(self, alpha=0.9, beta=0.5, gamma=0.1):
        super(MultiLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.ssim_loss = losses.SSIMLoss(window_size=7, reduction='mean')
        self.edge_loss = EdgeLoss()
        self.alpha = alpha  # Weight for Huber Loss
        self.beta = beta    # Weight for SSIM Loss
        self.gamma = gamma  # Weight for Edge Loss

    def forward(self, outputs, targets):
        loss_huber = self.huber_loss(outputs, targets)
        loss_ssim = self.ssim_loss(outputs, targets)
        loss_edge = self.edge_loss(outputs, targets)
        total_loss = self.alpha * loss_huber + self.beta * loss_ssim + self.gamma * loss_edge
        return total_loss, loss_huber, loss_ssim, loss_edge

# Initialize the loss function with specified weights
# criterion = MultiLoss(alpha=0.5, beta=0.3, gamma=0.2)
criterion = MultiLoss(alpha=HUBER, beta=SSIM, gamma=EDGE)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder().to(device)
optimizer = optim.AdamW(autoencoder.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize TensorBoard writer with date and time in the run name
log_dir = f'/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/runs/train_autoencoder_{RUNTYPE}_{current_time}'
writer = SummaryWriter(log_dir=log_dir)

def calculate_accuracy_bandwise(outputs, targets):
    """
    Calculate accuracy band by band.
    
    Args:
        outputs: Predicted tensor of shape (batch_size, num_bands, height, width).
        targets: Ground truth tensor of shape (batch_size, num_bands, height, width).
    
    Returns:
        Mean accuracy across all bands.
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    batch_size, num_bands, _, _ = outputs.shape
    band_accuracies = []
    
    for band in range(num_bands):
        # Get the predictions and targets for the current band
        output_band = outputs[:, band, :, :]
        target_band = targets[:, band, :, :]
        
        # Clip values to [0, 1] to avoid issues with rounding
        output_band = np.clip(output_band, 0, 1)
        target_band = np.clip(target_band, 0, 1)
        
        # Calculate accuracy for the current band
        band_accuracy = np.mean(np.isclose(output_band, target_band, atol=0.1))
        band_accuracies.append(band_accuracy)
    
    # Return the mean accuracy across all bands
    return np.mean(band_accuracies)

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# SSIM Calculation using Kornia
def calculate_ssim(img1, img2):
    return ssim(img1, img2, window_size=7)

############################# TRAINING ################################

# Training loop
best_val_accuracy = 0.0
for epoch in range(EPOCHS):
    autoencoder.train()
    running_loss = 0.0
    running_huber_loss = 0.0
    running_ssim_loss = 0.0
    running_edge_loss = 0.0
    running_acc = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for images in tqdm(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(images)

        # Compute the MultiLoss components
        loss, loss_huber, loss_ssim, loss_edge = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        # Track all loss components
        running_loss += loss.item() * images.size(0)
        running_huber_loss += loss_huber.item() * images.size(0)
        running_ssim_loss += loss_ssim.item() * images.size(0)
        running_edge_loss += loss_edge.item() * images.size(0)

        # Track accuracy (band-wise)
        running_acc += calculate_accuracy_bandwise(outputs, images) * images.size(0)

        # Track PSNR and SSIM
        running_psnr += calculate_psnr(outputs, images).item() * images.size(0)
        running_ssim += calculate_ssim(outputs, images).mean().item() * images.size(0)

    # Calculate average metrics for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_huber_loss = running_huber_loss / len(train_loader.dataset)
    epoch_ssim_loss = running_ssim_loss / len(train_loader.dataset)
    epoch_edge_loss = running_edge_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_psnr = running_psnr / len(train_loader.dataset)
    epoch_ssim = running_ssim / len(train_loader.dataset)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Huber: {epoch_huber_loss:.4f}, '
          f'SSIM: {epoch_ssim_loss:.4f}, Edge: {epoch_edge_loss:.4f}, Accuracy: {epoch_acc:.4f}, PSNR: {epoch_psnr:.4f}, SSIM: {epoch_ssim:.4f}')

    # scheduler.step()

    # current_lr = scheduler.get_last_lr()[0]
    # print(f"Current learning rate: {current_lr:.6f}")
    # writer.add_scalar('Learning Rate', current_lr, epoch)

    # Log training metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Loss/train_huber', epoch_huber_loss, epoch)
    writer.add_scalar('Loss/train_ssim', epoch_ssim_loss, epoch)
    writer.add_scalar('Loss/train_edge', epoch_edge_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    writer.add_scalar('PSNR/train', epoch_psnr, epoch)
    writer.add_scalar('SSIM/train', epoch_ssim, epoch)

    ############################# VALIDATION ################################

    # Validation phase
    autoencoder.eval()
    val_loss = 0.0
    val_huber_loss = 0.0
    val_ssim_loss = 0.0
    val_edge_loss = 0.0
    val_acc = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for images in tqdm(val_loader):
            images = images.to(device)
            outputs = autoencoder(images)

            # Compute the MultiLoss components for validation
            loss, loss_huber, loss_ssim, loss_edge = criterion(outputs, images)

            val_loss += loss.item() * images.size(0)
            val_huber_loss += loss_huber.item() * images.size(0)
            val_ssim_loss += loss_ssim.item() * images.size(0)
            val_edge_loss += loss_edge.item() * images.size(0)
            val_acc += calculate_accuracy_bandwise(outputs, images) * images.size(0)
            val_psnr += calculate_psnr(outputs, images).item() * images.size(0)
            val_ssim += calculate_ssim(outputs, images).mean().item() * images.size(0)


    # Calculate average metrics for validation
    val_loss /= len(val_loader.dataset)
    val_huber_loss /= len(val_loader.dataset)
    val_ssim_loss /= len(val_loader.dataset)
    val_edge_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_psnr /= len(val_loader.dataset)
    val_ssim /= len(val_loader.dataset)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Validation Loss: {val_loss:.4f}, Validation Huber: {val_huber_loss:.4f}, '
          f'Validation SSIM: {val_ssim_loss:.4f}, Validation Edge: {val_edge_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation PSNR: {val_psnr:.4f}, Validation SSIM: {val_ssim:.4f}')

    # Log validation metrics to TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/val_huber', val_huber_loss, epoch)
    writer.add_scalar('Loss/val_ssim', val_ssim_loss, epoch)
    writer.add_scalar('Loss/val_edge', val_edge_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('PSNR/val', val_psnr, epoch)
    writer.add_scalar('SSIM/val', val_ssim, epoch)

    # Save the best model based on validation accuracy
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_epoch = epoch + 1

        # Construct the best model path with epoch in the name
        best_model_path = os.path.join(models_dir, f'{RUNTYPE}_best_autoencoder_at{best_epoch}epochs_{EPOCHS}epochs.pth')

        # Remove any existing best autoencoder models to avoid clutter
        for file in os.listdir(models_dir):
            if file.startswith(f'{RUNTYPE}_best_autoencoder_at') and file.endswith('.pth'):
                os.remove(os.path.join(models_dir, file))

        # Save the current best model
        torch.save(autoencoder.state_dict(), best_model_path)
        print(f"Best autoencoder model saved with validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")

        # Construct the best encoder path with epoch in the name
        best_encoder_path = os.path.join(models_dir, f'{RUNTYPE}_best_autoencoder_encoderweights_at{best_epoch}epochs_{EPOCHS}epochs.pth')

        # Remove any existing best encoder weights to avoid clutter
        for file in os.listdir(models_dir):
            if file.startswith(f'{RUNTYPE}_best_autoencoder_encoderweights_at') and file.endswith('.pth'):
                os.remove(os.path.join(models_dir, file))

        # Save only the encoder weights of the best autoencoder
        encoder = Encoder().to(device)
        encoder.load_state_dict(autoencoder.encoder.state_dict())  # Extract encoder weights
        torch.save(encoder.state_dict(), best_encoder_path)
        print(f"Best encoder weights saved to: {best_encoder_path}")

    # # Save the final autoencoder model
    # final_autoencoder_model_path = os.path.join(models_dir, f'final_autoencoder.pth')
    # torch.save(autoencoder.state_dict(), final_autoencoder_model_path)
    # print(f"Final autoencoder model saved to {final_autoencoder_model_path}")

############################# TESTING #############################

# # Load the last saved model
# autoencoder.load_state_dict(torch.load(final_autoencoder_model_path))
# autoencoder.eval()

print("\nRunning final evaluation on test set...")

# Initialize metrics for test set
test_loss = 0.0
test_huber_loss = 0.0
test_ssim_loss = 0.0
test_edge_loss = 0.0
test_acc = 0.0
test_psnr = 0.0
test_ssim_value = 0.0

with torch.no_grad():
    for images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = autoencoder(images)

        # Compute the MultiLoss components for the test set
        loss, loss_huber, loss_ssim, loss_edge = criterion(outputs, images)

        test_loss += loss.item() * images.size(0)
        test_huber_loss += loss_huber.item() * images.size(0)
        test_ssim_loss += loss_ssim.item() * images.size(0)
        test_edge_loss += loss_edge.item() * images.size(0)
        test_acc += calculate_accuracy_bandwise(outputs, images) * images.size(0)
        test_psnr += calculate_psnr(outputs, images).item() * images.size(0)
        test_ssim_value += calculate_ssim(outputs, images).mean().item() * images.size(0)

# Calculate average metrics for the test set
test_loss /= len(test_loader.dataset)
test_huber_loss /= len(test_loader.dataset)
test_ssim_loss /= len(test_loader.dataset)
test_edge_loss /= len(test_loader.dataset)
test_acc /= len(test_loader.dataset)
test_psnr /= len(test_loader.dataset)
test_ssim_value /= len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Huber: {test_huber_loss:.4f}, '
        f'Test SSIM: {test_ssim_loss:.4f}, Test Edge: {test_edge_loss:.4f}, '
        f'Test Accuracy: {test_acc:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim_value:.4f}')

# # Log the test metrics to TensorBoard
# writer.add_scalar('Loss/test', test_loss, epoch)
# writer.add_scalar('Loss/test_huber', test_huber_loss, epoch)
# writer.add_scalar('Loss/test_ssim', test_ssim_loss ,epoch)
# writer.add_scalar('Loss/test_edge', test_edge_loss, epoch)
# writer.add_scalar('Accuracy/test', test_acc, epoch)
# writer.add_scalar('PSNR/test', test_psnr, epoch)
# writer.add_scalar('SSIM/test', test_ssim_value, epoch)

# Save the final autoencoder model after training completes
final_autoencoder_model_path = os.path.join(models_dir, f'{RUNTYPE}_final_autoencoder_{EPOCHS}epochs.pth')
torch.save(autoencoder.state_dict(), final_autoencoder_model_path)
print(f"Final autoencoder model saved to {final_autoencoder_model_path}")

# Close the TensorBoard writer
writer.close()

############################# VISUALIZATION #############################

# # Define best_model_path if not already defined
# best_model_path = os.path.join(models_dir, 'best_autoencoder.pth')

# final_autoencoder_model_path = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Sentinel/Autoencoder_for_pretraining/100percent/05_03_02_00001_015dropout_final_autoencoder_200epochs.pth"

# Load the autoencoder model
autoencoder.load_state_dict(torch.load(final_autoencoder_model_path, weights_only=True))
autoencoder.eval()

# # Test on a few images
# num = np.random.randint(0, len(test_dataset))
# test_img = test_dataset[num].unsqueeze(0).to(device)
# raw_image = raw_test_dataset[num]

# print("Raw image file:", raw_test_dataset[num])
# print("Processed image file:", test_dataset[num])

# with torch.no_grad():
#     pred = autoencoder(test_img)

# # # Apply contrast stretching
# # def contrast_stretch(image):
# #     p2, p98 = np.percentile(image, (2, 98))
# #     return np.clip((image - p2) / (p98 - p2), 0, 1)

# # Apply contrast stretching
# def contrast_stretch(image, brightness_factor=1.2):
#     image = (image - image.min()) / (image.max() - image.min())
    
#     # Apply contrast stretching per channel to maintain color balance
#     p2, p98 = np.percentile(image, (2, 98), axis=(0, 1))
#     image = np.clip((image - p2) / (p98 - p2), 0, 1)
    
#     # Increase brightness while maintaining color proportions
#     image = np.clip(image * brightness_factor, 0, 1)
    
#     return image

# # Convert raw image to RGB and apply contrast stretching
# raw_image_rgb = raw_image[:, :, [2, 1, 0]]
# raw_image_rgb = contrast_stretch(raw_image_rgb, brightness_factor=1.2)

# # Original (input) image
# original_image_rgb = test_img.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]

# # Reconstructed image
# reconstructed_image_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]

# # Compute error map (absolute difference)
# error_map = np.abs(original_image_rgb - reconstructed_image_rgb)
# # For visualization, we can take the mean across bands to get a single-channel error map
# error_map_gray = np.mean(error_map, axis=2)

# # Plot the images (now 4 subplots)
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# # Raw image
# axes[0].imshow(raw_image_rgb)
# axes[0].set_title("Raw (RGB)")
# axes[0].axis("off")

# # Original (input to autoencoder)
# axes[1].imshow(original_image_rgb)
# axes[1].set_title("Original (RGB)")
# axes[1].axis("off")

# # Reconstructed image (output from autoencoder)
# axes[2].imshow(reconstructed_image_rgb)
# axes[2].set_title("Reconstructed (RGB)")
# axes[2].axis("off")

# # Error map visualization
# im = axes[3].imshow(error_map_gray, cmap='viridis', aspect='auto')
# axes[3].set_title("Error Map")
# axes[3].axis("off")

# # Add a colorbar to interpret error map values
# fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label='Absolute Difference')

# plt.tight_layout(pad=1.0)

# # Save the figure
# num_images_str = NUM_IMAGES if NUM_IMAGES is not None else "all"
# output_path = os.path.join(
#     output_dir, f"{RUNTYPE}_autoencoder_reconstructed_{EPOCHS}epochs_{SUBSET}%subset.png"
# )
# plt.savefig(output_path)
# print(f"Image saved to {output_path}")

num_images = 30

for i in range(num_images):
    num = np.random.randint(0, len(test_dataset))
    test_img = test_dataset[num].unsqueeze(0).to(device)
    raw_image = raw_test_dataset[num]

    print(f"Processing image {i+1}/{num_images}")

    with torch.no_grad():
        pred = autoencoder(test_img)

    # Apply contrast stretching
    def contrast_stretch(image, brightness_factor=1.2):
        image = (image - image.min()) / (image.max() - image.min())
        
        # Apply contrast stretching per channel to maintain color balance
        p2, p98 = np.percentile(image, (2, 98), axis=(0, 1))
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
        
        # Increase brightness while maintaining color proportions
        image = np.clip(image * brightness_factor, 0, 1)
        
        return image

    # Convert raw image to RGB and apply contrast stretching
    raw_image_rgb = raw_image[:, :, [2, 1, 0]]
    raw_image_rgb = contrast_stretch(raw_image_rgb, brightness_factor=0.85)

    # Original (input) image
    original_image_rgb = test_img.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]

    # Reconstructed image
    reconstructed_image_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]

    # Compute error map (absolute difference)
    error_map = np.abs(original_image_rgb - reconstructed_image_rgb)
    # For visualization, we can take the mean across bands to get a single-channel error map
    error_map_gray = np.mean(error_map, axis=2)

    # Plot the images (now 4 subplots)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Raw image
    axes[0].imshow(raw_image_rgb)
    axes[0].set_title("Raw (RGB)")
    axes[0].axis("off")

    # Original (input to autoencoder)
    axes[1].imshow(original_image_rgb)
    axes[1].set_title("Original (RGB)")
    axes[1].axis("off")

    # Reconstructed image (output from autoencoder)
    axes[2].imshow(reconstructed_image_rgb)
    axes[2].set_title("Reconstructed (RGB)")
    axes[2].axis("off")

    # Error map visualization
    im = axes[3].imshow(error_map_gray, cmap='viridis', aspect='auto')
    axes[3].set_title("Error Map")
    axes[3].axis("off")

    # Add a colorbar to interpret error map values
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label='Absolute Difference')

    plt.tight_layout(pad=1.0)

    # Save the figure
    output_path = os.path.join(
        output_dir, f"{RUNTYPE}_autoencoder_reconstructed_{EPOCHS}epochs_{SUBSET}%subset_img{i+1}.png"
    )
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Image {i+1}/{num_images} saved to {output_path}")

############################# SAVE WEIGHTS FOR UNET #############################

autoencoder.load_state_dict(torch.load(final_autoencoder_model_path))
autoencoder.eval()

# Extract weights only for the encoder part of the Autoencoder
encoder = Encoder().to(device)
encoder.load_state_dict(autoencoder.encoder.state_dict())

# Save encoder weights for future comparison
encoder_path = os.path.join(models_dir, f'{RUNTYPE}_final_autoencoder_encoderweights_{EPOCHS}epochs.pth')
torch.save(encoder.state_dict(), encoder_path)
print(f"Encoder weights saved to {encoder_path}")

# # Check the output of encoder_model on a test image
# with torch.no_grad():
#     temp_img_encoded, _ = encoder(test_img)

# # Now define a U-Net with the same encoder part as our autoencoder
# unet = Unet().to(device)

# # Set weights to encoder part of the U-Net
# unet.encoder.load_state_dict(encoder.state_dict())

# # Save the U-Net model
# unet_path = os.path.join(models_dir, f'unet_model_withpretrainedweights_{EPOCHS}epochs.pth')
# torch.save(unet.state_dict(), unet_path)
# print(f"U-Net model saved to {unet_path}")

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time in hours, minutes, and seconds
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total time taken to run script: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
