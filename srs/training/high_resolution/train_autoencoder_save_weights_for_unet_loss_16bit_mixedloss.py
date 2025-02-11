import os
import random
import time
from datetime import datetime

import kornia.losses as losses
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kornia.metrics import ssim
from matplotlib import pyplot as plt
from models import Autoencoder, Encoder, Unet
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import equalize, to_pil_image, to_tensor
from tqdm import tqdm

# Record the start time
start_time = time.time()

############################# SETTING UP THE PARAMETERS #############################

SIZE = 256
BATCH_SIZE = 8
EPOCHS = 1
NUM_IMAGES = None
SEED = 42

RUNTYPE = 'histogram_equal'

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

############################# LOADING THE DATA #############################

# Ensure the saved_images directory exists
output_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/saved_images_reconstruction"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure the models directory exists
models_dir = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load data for U-net training
train_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Wetlands_Blackshark/Wetlands_trainvaltest/images_train'
val_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Wetlands_Blackshark/Wetlands_trainvaltest/images_val'
test_images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Wetlands_Blackshark/Wetlands_trainvaltest/images_test'

############################# DEFINE HELPER AND DATA LOADING FUNCTIONS #############################

def save_combined_pixel_distribution(image, prefix='', suffix=''):
    """
    Create a combined histogram plot for all bands of the input image with descriptive band names.

    Args:
        image (numpy.ndarray): Input image in HWC (Height x Width x Channels) format.
        prefix (str): Optional prefix for the output file name.
        suffix (str): Optional suffix for the output file name.
    """

    # Ensure the output directory exists
    output_dir = '/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/pixeldistribution'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the min and max pixel values dynamically
    pixel_min = image.min()
    pixel_max = image.max()

    # Number of bands (assuming HWC format)
    bands = image.shape[2]

    # Band names corresponding to indices
    band_names = ["B1 (Red)", "B2 (Green)", "B3 (Blue)", "B4 (NIR)"]

    # Colors for each band
    colors = ['red', 'green', 'blue', '#1abc9c']  # Colors for Red, Green, Blue, and NIR

    # Plot histograms for the selected bands
    plt.figure(figsize=(10, 5))
    for i in range(min(bands, len(band_names))):  # Ensure we don't exceed the available bands
        plt.hist(image[:, :, i].ravel(), bins=100, range=(pixel_min, pixel_max), alpha=0.6,
                 color=colors[i % len(colors)], label=band_names[i])

    plt.title('Histogram Equalization (Image)')
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

def apply_histogram_equalization_16bit(image):
    """
    Perform histogram equalization on a multi-channel 16-bit image.
    
    Args:
        image (numpy.ndarray): Input image of shape (H, W, C) with pixel values in [0, 65535].
    
    Returns:
        numpy.ndarray: Histogram-equalized image of same shape as input.
    """
    MAX_16BIT = 65535
    equalized_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):  # Process each channel independently
        channel = image[:, :, c]
        
        # Compute the histogram and cumulative distribution function (CDF)
        hist, bins = np.histogram(channel.flatten(), bins=MAX_16BIT + 1, range=(0, MAX_16BIT))
        cdf = hist.cumsum()
        cdf_normalized = cdf * (MAX_16BIT / cdf[-1])  # Normalize CDF to range [0, MAX_16BIT]
        
        # Use linear interpolation of the CDF to find new pixel values
        equalized_image[:, :, c] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)
    
    return equalized_image.astype(np.float32)  # Keep as float32 for further processing

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None, raw=False):
        self.image_dir = image_dir
        self.image_files = image_files
        self.transform = transform
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
            # Apply contrast stretching
            print(f"[Image {idx}] Raw: min={image.min()}, max={image.max()}")
            p2, p98 = np.percentile(image, (2, 98))
            image = np.clip((image - p2) / (p98 - p2), 0, 1)
            return image

        # For the first 5 images, save the pixel distribution
        if idx < 100:
            save_combined_pixel_distribution(image, prefix=f"image_{idx}_", suffix="_before normalization")

        # # print pixel range of raw image before transform per band
        # for band in range(image.shape[2]):
        #     print(f"[Image {idx}] before normalization Band {band}: min={image[:, :, band].min()}, max={image[:, :, band].max()}")

        # Perform histogram equalization
        # image = histogram_equalization_16bit(image)

        # Apply histogram equalization
        image = apply_histogram_equalization_16bit(image)

        # Normalize the image to [0, 1]
        image = image / 65535.0

        # # Normalize to [0, 1] for model input
        # image = image / 10000.0
        # image = image / 23480.0  # Normalize to [0, 1]

        # For the first 5 images, save the pixel distribution
        if idx < 100:
            save_combined_pixel_distribution(image, prefix=f"image_{idx}_", suffix="_after normalization")

        # # print pixel range of raw image before transform per band
        # for band in range(image.shape[2]):
        #     print(f"[Image {idx}] before transformation Band {band}: min={image[:, :, band].min()}, max={image[:, :, band].max()}")

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
        self.huber_loss = nn.HuberLoss(delta=0.1)
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
criterion = MultiLoss(alpha=0.6, beta=0.4, gamma=0.2)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize TensorBoard writer with date and time in the run name
log_dir = f'/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/runs/train_autoencoder_{current_time}'
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
        best_model_path = os.path.join(models_dir, 'best_autoencoder.pth')
        torch.save(autoencoder.state_dict(), best_model_path)
        print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")

# # Save the final autoencoder model
# final_autoencoder_model_path = os.path.join(models_dir, f'final_autoencoder.pth')
# torch.save(autoencoder.state_dict(), final_autoencoder_model_path)
# print(f"Final autoencoder model saved to {final_autoencoder_model_path}")

############################# TESTING #############################

# # Load the last saved model
# autoencoder.load_state_dict(torch.load(final_autoencoder_model_path))
# autoencoder.eval()

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

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Test Loss: {test_loss:.4f}, Test Huber: {test_huber_loss:.4f}, '
          f'Test SSIM: {test_ssim_loss:.4f}, Test Edge: {test_edge_loss:.4f}, '
          f'Test Accuracy: {test_acc:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim_value:.4f}')

    # Log the test metrics to TensorBoard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Loss/test_huber', test_huber_loss, epoch)
    writer.add_scalar('Loss/test_ssim', test_ssim_loss, epoch)
    writer.add_scalar('Loss/test_edge', test_edge_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('PSNR/test', test_psnr, epoch)
    writer.add_scalar('SSIM/test', test_ssim_value, epoch)

# Save the final autoencoder model after training completes
final_autoencoder_model_path = os.path.join(models_dir, f'final_autoencoder.pth')
torch.save(autoencoder.state_dict(), final_autoencoder_model_path)
print(f"Final autoencoder model saved to {final_autoencoder_model_path}")

# Close the TensorBoard writer
writer.close()

############################# VISUALIZATION #############################

# # Define best_model_path if not already defined
# best_model_path = os.path.join(models_dir, 'best_autoencoder.pth')

# final_autoencoder_model_path = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/models/final_autoencoder.pth"

# Load the autoencoder model
autoencoder.load_state_dict(torch.load(final_autoencoder_model_path, weights_only=True))
autoencoder.eval()

# num = np.random.randint(0, len(test_dataset))
num = 40
raw_test_image = raw_test_dataset[num]
test_image = test_dataset[num].unsqueeze(0).to(device)

with torch.no_grad():
    pred = autoencoder(test_image)

# Original (input) image
original_image_rgb = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]

# Reconstructed image
reconstructed_image_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]

# Compute error map (absolute difference)
error_map = np.abs(original_image_rgb - reconstructed_image_rgb)
# For visualization, we can take the mean across bands to get a single-channel error map
error_map_gray = np.mean(error_map, axis=2)

# Plot the images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figure size as needed

# Apply contrast stretching
def contrast_stretch(image, brightness_factor=1.2):
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply contrast stretching per channel to maintain color balance
    p2, p98 = np.percentile(image, (2, 98), axis=(0, 1))
    image = np.clip((image - p2) / (p98 - p2), 0, 1)
    
    # Increase brightness while maintaining color proportions
    image = np.clip(image * brightness_factor, 0, 1)
    
    return image

# Ensure raw_test_image is properly normalized and has 3 channels
raw_test_image = raw_test_image.astype(np.float32)[:, :, :3]
# raw_test_image = (raw_test_image - raw_test_image.min()) / (raw_test_image.max() - raw_test_image.min())
raw_test_image = contrast_stretch(raw_test_image, brightness_factor=1.2)
axes[0].imshow(raw_test_image, vmin=0, vmax=1)
axes[0].set_title('Raw (RGB)')
axes[0].axis('off')

original_image = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]
axes[1].imshow(original_image)
axes[1].set_title('Original (RGB)')
axes[1].axis('off')

reconstructed_image_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]
axes[2].imshow(reconstructed_image_rgb)
axes[2].set_title('Original (RGB)')
axes[2].axis('off')

# Error map visualization
im = axes[3].imshow(error_map_gray, cmap='viridis', aspect='auto')
axes[3].set_title("Error Map")
axes[3].axis("off")

# Add a colorbar to interpret error map values
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label='Absolute Difference')

plt.tight_layout(pad=1.0)

# Save the figure
num_images_str = NUM_IMAGES if NUM_IMAGES is not None else "all"
output_path = os.path.join(
    output_dir, f"{RUNTYPE}_autoencoder_reconstructed_{EPOCHS}epochs.png"
)
plt.savefig(output_path)
print(f"Image saved to {output_path}")

# image = raw_test_dataset[40]
# output_file = os.path.join(output_dir, f'raw_image_{40}.png')
# # Print the pixel min and max values of the saved image
# print(f"[Image {40}] Raw: min={image.min()}, max={image.max()}")
# plt.imsave(output_file, image)
# print(f"Image saved to {output_file}")

# # Define number of images to process
# num_images = 40

# # Loop over 20 images
# for i in range(20, num_images):
#     num = np.random.randint(0, len(test_dataset))  # Randomly select an image
#     raw_test_image = raw_test_dataset[num]
#     test_image = test_dataset[num].unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred = autoencoder(test_image)

#     # Original (input) image
#     original_image_rgb = test_image.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]

#     # Reconstructed image
#     reconstructed_image_rgb = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, :3]

#     # Compute error map (absolute difference)
#     error_map = np.abs(original_image_rgb - reconstructed_image_rgb)
#     error_map_gray = np.mean(error_map, axis=2)

#     # Plot the images
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))

#     # Contrast stretching function
#     def contrast_stretch(image):
#         p2, p98 = np.percentile(image, (2, 98))
#         return np.clip((image - p2) / (p98 - p2), 0, 1)

#     def normalize_and_brighten(image, brightness_factor=1.2):
#         """Normalize image and increase brightness by scaling pixel values"""
#         image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
#         image = np.clip(image * brightness_factor, 0, 1)  # Scale brightness
#         return image

#     # Ensure raw_test_image is properly normalized
#     raw_test_image = raw_test_image.astype(np.float32)[:, :, :3]
#     raw_test_image = normalize_and_brighten(raw_test_image)
#     axes[0].imshow(raw_test_image, vmin=0, vmax=1)
#     axes[0].set_title('Raw (RGB)')
#     axes[0].axis('off')

#     axes[1].imshow(original_image_rgb)
#     axes[1].set_title('Original (RGB)')
#     axes[1].axis('off')

#     axes[2].imshow(reconstructed_image_rgb)
#     axes[2].set_title('Reconstructed (RGB)')
#     axes[2].axis('off')

#     im = axes[3].imshow(error_map_gray, cmap='viridis', aspect='auto')
#     axes[3].set_title("Error Map")
#     axes[3].axis("off")

#     fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label='Absolute Difference')

#     plt.tight_layout(pad=1.0)

#     # Save the figure
#     output_path = os.path.join(
#         output_dir, f"{RUNTYPE}_autoencoder_reconstructed_{EPOCHS}epochs_img{i}.png"
#     )
#     plt.savefig(output_path)
#     plt.close(fig)
#     print(f"Image {i+1}/{num_images} saved to {output_path}")


############################# SAVE WEIGHTS FOR UNET #############################

autoencoder.load_state_dict(torch.load(final_autoencoder_model_path))
autoencoder.eval()

# Extract weights only for the encoder part of the Autoencoder
encoder = Encoder().to(device)
encoder.load_state_dict(autoencoder.encoder.state_dict())

# Save encoder weights for future comparison
encoder_path = os.path.join(models_dir, 'pretrained_encoder_weights.pth')
torch.save(encoder.state_dict(), encoder_path)
print(f"Encoder weights saved to {encoder_path}")

# Check the output of encoder_model on a test image
with torch.no_grad():
    temp_img_encoded, _ = encoder(test_image)

# Now define a U-Net with the same encoder part as our autoencoder
unet = Unet().to(device)

# Set weights to encoder part of the U-Net
unet.encoder.load_state_dict(encoder.state_dict())

# Save the U-Net model
unet_path = os.path.join(models_dir, f'unet_model_withpretrainedweights_{EPOCHS}epochs.pth')
torch.save(unet.state_dict(), unet_path)
print(f"U-Net model saved to {unet_path}")

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time in hours, minutes, and seconds
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total time taken to run script: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")