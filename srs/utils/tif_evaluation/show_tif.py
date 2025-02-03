import os
import rasterio
from matplotlib import pyplot as plt
import numpy as np

def save_tiff_as_png(input_file, output_file):
    """
    Reads a TIFF image file and saves it as a PNG file.

    Args:
        input_file (str): The path to the input TIFF file.
        output_file (str): The path to the output PNG file.
    """
    # Use rasterio to read the TIFF file
    with rasterio.open(input_file) as src:
        image = src.read()  # Read all bands
        image = image.transpose(1, 2, 0)  # Transpose to HWC format

    # Apply contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip((image - p2) / (p98 - p2), 0, 1)

    # Save the image to a PNG file
    plt.imsave(output_file, image)
    print(f"Image saved to {output_file}")

# Example usage

input_file = "/projects/0/prjs1235/Satellietdataportaal_data/images_val/20230907_105113_SVNEO_Lauwersoog_tile_1_55.tif"
output_file = "/home/egmelich/SatelliteMAE/Autoencoder_Unet_Satellietdataportaal/saved_images/20230907_105113_SVNEO_Lauwersoog_tile_1_55.png"

save_tiff_as_png(input_file, output_file)