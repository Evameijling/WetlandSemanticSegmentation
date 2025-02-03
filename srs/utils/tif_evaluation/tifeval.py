import os
import sys
import argparse
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process .tif files and generate subfigures and GIF.")
    parser.add_argument("band", type=str, choices=["R", "G", "B", "NIR"], help="Band to process (R, G, B, or NIR)")
    return parser.parse_args()

def format_date(date_str):
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    return f'{year}-{month}-{day}'

def find_global_min_max(analytics_files, band):
    global_min = float('inf')
    global_max = float('-inf')

    for analytics_file in analytics_files:
        with rasterio.open(analytics_file) as src:
            if band == "R":
                band_data = src.read(3)
            elif band == "G":
                band_data = src.read(2)
            elif band == "B":
                band_data = src.read(1)
            elif band == "NIR":
                band_data = src.read(4)

            local_min = np.nanmin(band_data)
            local_max = np.nanmax(band_data)

            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max

    return global_min, global_max

def process_images(analytics_tif, udm2_tif, output_folder, band, vmin, vmax):
    date = os.path.basename(analytics_tif).split('_')[0]

    # Check if both files exist
    if not os.path.exists(analytics_tif):
        print(f"Analytics file for date {date} not found at {analytics_tif}")
        return
    if not os.path.exists(udm2_tif):
        print(f"UDM2 file for date {date} not found at {udm2_tif}")
        return

    # Read the Analytics file
    with rasterio.open(analytics_tif) as src:
        if band == "R":
            band_data = src.read(3)  # Assuming band 3 is the Red band
            cmap = 'Reds'
        elif band == "G":
            band_data = src.read(2)  # Assuming band 2 is the Green band
            cmap = 'Greens'
        elif band == "B":
            band_data = src.read(1)  # Assuming band 1 is the Blue band
            cmap = 'Blues'
        elif band == "NIR":
            band_data = src.read(4)  # Assuming band 4 is the NIR band
            cmap = 'viridis'

    # Read the UDM2 file (assumed to be a single band)
    with rasterio.open(udm2_tif) as src:
        udm2_mask = src.read(1)

    # Apply the UDM2 mask: set usable areas to NaN
    masked_unusable = np.where(udm2_mask == 0, band_data, np.nan)

    # Create a combined image showing unusable spots
    combined_image = band_data.copy()
    combined_image[udm2_mask == 0] = 0  # Mark unusable areas in the combined image

    # Format the date for the title
    formatted_date = format_date(date)

    # Create a figure with subplots using gridspec
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # Subimage 1: Band image
    ax0 = plt.subplot(gs[0])
    im0 = ax0.imshow(band_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax0.set_title(f'{band} Band ({formatted_date})')
    ax0.axis('off')

    # Subimage 2: Masked unusable areas
    ax1 = plt.subplot(gs[1])
    im1 = ax1.imshow(masked_unusable, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f'Masked Unusable Areas ({formatted_date})')
    ax1.axis('off')

    # Subimage 3: Combined image showing unusable spots
    ax2 = plt.subplot(gs[2])
    im2 = ax2.imshow(band_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.imshow(np.ma.masked_where(udm2_mask != 0, udm2_mask), cmap='gray', alpha=0.5)
    ax2.set_title(f'Combined Image with Unusable Spots ({formatted_date})')
    ax2.axis('off')

    # Add a vertical color bar to the right of the subplots
    cbar_ax = plt.subplot(gs[3])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('Intensity Values (units)')

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, f'{date}_{band}_output.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved output for date {date} to {output_path}")

    return output_path

def main():
    args = parse_arguments()
    band = args.band

    # Folder containing the .tif files
    folder_path = '../../data/planet_data/'

    # Create the output folder for the band if it doesn't exist
    output_folder = os.path.join('../../outputs/planet_data', band)
    os.makedirs(output_folder, exist_ok=True)

    # Get list of analytics files in the folder
    analytics_files = glob.glob(os.path.join(folder_path, '*_3B_AnalyticMS_SR_harmonized_clip.tif'))

    # Find global min and max intensity values
    vmin, vmax = find_global_min_max(analytics_files, band)

    # List to store output paths for the GIF
    output_paths = []

    # Process images for each analytics file
    for analytics_file in analytics_files:
        date_part = os.path.basename(analytics_file).split('_')[0]
        udm2_file_pattern = os.path.join(folder_path, f'{date_part}_*_3B_udm2_clip.tif')
        udm2_files = glob.glob(udm2_file_pattern)

        if udm2_files:
            udm2_file = udm2_files[0]  # Assuming there is only one matching UDM2 file
            output_path = process_images(analytics_file, udm2_file, output_folder, band, vmin, vmax)
            if output_path:
                output_paths.append(output_path)
        else:
            print(f"No matching UDM2 file found for date {date_part}")

    # Create GIF from the output images
    if output_paths:
        images = [imageio.imread(path) for path in sorted(output_paths)]
        gif_output_path = os.path.join(output_folder, f'evolution_over_time_{band}.gif')
        imageio.mimsave(gif_output_path, images, duration=1000)
        print(f"GIF saved at {gif_output_path}")
    else:
        print("No images were processed, GIF was not created.")

if __name__ == "__main__":
    main()