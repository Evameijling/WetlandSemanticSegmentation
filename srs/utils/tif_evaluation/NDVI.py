import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def calculate_and_save_ndvi_with_colormap(input_tiff_path, output_file_path):
    """
    Calculate NDVI from a satellite image and save it as a color-mapped image with an overlaid legend.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    print(f"\nOpening {input_tiff_path} with Rasterio ...")
    with rasterio.open(input_tiff_path) as src:
        # Assume NIR is band 4 and RED is band 3 (change if different)
        nir_band = src.read(4).astype(np.float32)  # Near-Infrared
        red_band = src.read(3).astype(np.float32)  # Red

        print(f"NIR and RED bands loaded. Shape: {nir_band.shape}")

        # Avoid division by zero
        denominator = (nir_band + red_band)
        denominator[denominator == 0] = np.nan  # Set to NaN to avoid errors

        # Calculate NDVI
        ndvi = (nir_band - red_band) / denominator

    # Normalize NDVI to [0, 1] for color mapping (-1 to 1 becomes 0 to 1)
    ndvi_normalized = (ndvi + 1) / 2

    # Define a colormap: red (low NDVI) to green (high NDVI)
    colormap = LinearSegmentedColormap.from_list("ndvi_colormap", ["red", "yellow", "green"], N=256)

    # Create the figure and add NDVI with the color map
    fig, ax = plt.subplots(figsize=(ndvi_normalized.shape[1] / 100, ndvi_normalized.shape[0] / 100), dpi=100)
    cax = ax.imshow(ndvi_normalized, cmap=colormap)
    ax.axis("off")  # Turn off axes for better visualization

    # Add a colorbar directly onto the image
    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label("NDVI", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    print(f"Saving colorized NDVI with legend -> {output_file_path}")
    fig.savefig(output_file_path, format='PNG', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Colorized NDVI image with legend saved at {output_file_path}.")

def main():
    # Folder containing input TIFFs
    input_folder = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery'  # Update with your folder
    # Folder to save output NDVI images
    output_folder = '/home/egmelich/SatelliteMAE/NDVI'  # Update with your folder

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of all .tif files in the input folder
    tiff_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))

    print(f"Found {len(tiff_files)} TIFF files in {input_folder}.")

    for idx, tif_path in enumerate(tiff_files, start=1):
        # Get just the filename without extension
        base_name = os.path.splitext(os.path.basename(tif_path))[0]

        # Construct output filename
        output_file_path = os.path.join(
            output_folder,
            f"{base_name}_NDVI_colormap_legend.png"
        )

        print(f"\nProcessing file {idx}/{len(tiff_files)}: {os.path.basename(tif_path)}")
        
        # Call the NDVI calculation function
        calculate_and_save_ndvi_with_colormap(
            input_tiff_path=tif_path,
            output_file_path=output_file_path
        )

    print("\nAll NDVI images processed.")

if __name__ == '__main__':
    main()
