import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np

# Path to your TIFF file
tif_path = '/projects/0/prjs1235/data_for_keras_aug/train_images/train/GF2_PMS1__L1A0000647770-MSS1_310.tif'

# Open the TIFF file
with rasterio.open(tif_path) as dataset:
    # Read the first three bands (if available) for visualization (e.g., RGB)
    # You can modify the band indices (e.g., dataset.read(4)) to display specific bands
    red = dataset.read(1)  # Band 1
    green = dataset.read(2)  # Band 2
    blue = dataset.read(3)  # Band 3

    # Stack the bands to create an RGB image
    rgb_image = (rasterio.plot.reshape_as_image([red, green, blue]) / red.max())  # Normalize for visualization

    # Plot and save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.axis('off')
    output_path = f'output_{tif_path}.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

print(f"Image saved to {output_path}")