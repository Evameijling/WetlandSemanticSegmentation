import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# Directories
satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_selectedbands_cuts'
mask_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmergedmasks_cuts'
output_dir = '/home/egmelich/SatelliteMAE/Satellietdataportaal/SDPcombined'

# Define the color mapping and labels for each class
color_mapping = {
    0: (0, 0, 0),        # Negative (default) - Black
    1: (83, 251, 84),    # Grass & Farmland - #53FB54
    2: (224, 255, 6),    # Reed & Rough - #E0FF06
    3: (255, 214, 0),    # Shrubs - #FFD600
    4: (20, 102, 0),     # Forrest - #146600
    5: (0, 195, 206),    # Water - #00C3CE
    6: (255, 0, 0)       # Build - #FF0000
}

labels = {
    0: "Negative",
    1: "Grass & Farmland",
    2: "Reed & Rough",
    3: "Shrubs",
    4: "Forrest",
    5: "Water",
    6: "Build"
}

def process_files(satellite_path, mask_path, output_dir):
    # Output path for the combined PNG image
    output_png_path = os.path.join(output_dir, os.path.basename(satellite_path).replace('.tif', '_combined.png'))

    # Open the satellite image
    with rasterio.open(satellite_path) as sat_dataset:
        satellite_image = sat_dataset.read([1, 2, 3])  # Assuming RGB bands are 1, 2, 3

    # Open the mask image
    with rasterio.open(mask_path) as mask_dataset:
        mask = mask_dataset.read(1)

    # Create a colour-mapped array for the mask
    color_mapped_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_values = np.unique(mask)
    for value in unique_values:
        if value in color_mapping:
            mask_pixels = mask == value
            color_mapped_mask[mask_pixels] = color_mapping[value]

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the satellite image
    axes[0].imshow(np.moveaxis(satellite_image, 0, -1))
    axes[0].set_title('Satellite Image')
    axes[0].axis('off')

    # Plot the mask image
    axes[1].imshow(color_mapped_mask)
    axes[1].set_title('Mask Image')
    axes[1].axis('off')

    # Add a legend to the mask image
    legend_elements = [Patch(facecolor=np.array(color_mapping[value])/255, edgecolor='w', label=labels[value])
                       for value in unique_values if value in color_mapping]
    axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Save the combined image as PNG
    plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved combined image: {output_png_path}")

def find_and_process_file(satellite_filename, satellite_dir, mask_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full path to the satellite file
    satellite_path = os.path.join(satellite_dir, satellite_filename)
    
    if not os.path.exists(satellite_path):
        print(f"Satellite file {satellite_filename} does not exist in {satellite_dir}")
        return

    # Extract base name and tile info from the satellite filename
    base_name = '_'.join(satellite_filename.split('_')[:2])
    print(f"Base name: {base_name}")
    tile_info = '_'.join(satellite_filename.split('_')[-3:])
    print(f"Tile info: {tile_info}")

       # Find the corresponding mask file
    mask_filename = None
    for filename in os.listdir(mask_dir):
        if base_name in filename and tile_info in filename:
            mask_filename = filename
            break

    if mask_filename:
        mask_path = os.path.join(mask_dir, mask_filename)
        print(f"Found mask file: {mask_filename}")
        process_files(satellite_path, mask_path, output_dir)
    else:
        print(f"Corresponding mask file not found for {satellite_filename}")

# Example usage
satellite_filename = '20230430_104233_PNEO-04_1_1_30cm_RD_8bit_RGB_Sliedrecht_clipped_tile_127_145.tif'
find_and_process_file(satellite_filename, satellite_dir, mask_dir, output_dir)