# import os
# import rasterio
# from rasterio.windows import Window
# from rasterio.windows import transform as window_transform
# import numpy as np
# from PIL import Image

# # Input directories for satellite images and corresponding masks
# satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles'
# mask_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled"

# # Output directories for cut pieces
# output_satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/images_cuts_S2'
# output_mask_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_cuts_S2'

# # Create output directories if they don't exist
# os.makedirs(output_satellite_dir, exist_ok=True)
# os.makedirs(output_mask_dir, exist_ok=True)

# # Tile size
# tile_size = 256

# def cut_images_into_pieces(satellite_path, mask_path, satellite_output_dir, mask_output_dir, tile_size, check_black_threshold=0.05):
#     """
#     Cuts a satellite image and mask into smaller tiles and saves them if they meet the threshold for black pixels.

#     Args:
#         satellite_path (str): Path to the satellite image.
#         mask_path (str): Path to the corresponding mask.
#         satellite_output_dir (str): Directory to save the resulting satellite tiles.
#         mask_output_dir (str): Directory to save the resulting mask tiles.
#         tile_size (int): Size of each tile (assumed square).
#         check_black_threshold (float): Maximum allowable fraction of black pixels (0-1) in a tile.
#     """
#     with rasterio.open(satellite_path) as src_sat:
#         # Ensure the satellite and mask dimensions match
#         with Image.open(mask_path) as mask_img:
#             mask_array = np.array(mask_img.convert('L'))  # Convert mask to grayscale

#         assert src_sat.width == mask_array.shape[1] and src_sat.height == mask_array.shape[0], \
#             f"Mismatch in dimensions for {satellite_path} and {mask_path}"

#         meta_sat = src_sat.meta.copy()

#         # Calculate the number of tiles in each dimension
#         n_tiles_x = src_sat.width // tile_size
#         n_tiles_y = src_sat.height // tile_size

#         for i in range(n_tiles_x):
#             for j in range(n_tiles_y):
#                 # Calculate the window for the current tile
#                 window = Window(i * tile_size, j * tile_size, tile_size, tile_size)
#                 sat_tile = src_sat.read(window=window)
#                 mask_tile = mask_array[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size]

#                 # Check the percentage of black pixels in the mask
#                 black_pixels = (mask_tile == 5).sum()
#                 total_pixels = mask_tile.size
#                 black_fraction = black_pixels / total_pixels

#                 if black_fraction > check_black_threshold:
#                     print(f"Skipping tile ({i}, {j}) due to invalid pixels ({black_fraction:.2%}) in the mask.")
#                     continue

#                 # Update metadata for the tiles
#                 meta_sat.update({
#                     'height': tile_size,
#                     'width': tile_size,
#                     'transform': window_transform(window, src_sat.transform)
#                 })

#                 # Save the satellite tile
#                 sat_tile_path = os.path.join(satellite_output_dir, f"{os.path.splitext(os.path.basename(satellite_path))[0]}_tile_{i}_{j}.tif")
#                 with rasterio.open(sat_tile_path, 'w', **meta_sat) as dst_sat:
#                     dst_sat.write(sat_tile)

#                 # Save the mask tile
#                 mask_tile_path = os.path.join(mask_output_dir, f"{os.path.splitext(os.path.basename(mask_path))[0]}_tile_{i}_{j}.png")
#                 mask_tile_image = Image.fromarray(mask_tile)
#                 mask_tile_image.save(mask_tile_path)

# # Process files in the directories
# for satellite_filename in os.listdir(satellite_dir):
#     if satellite_filename.endswith('.tif'):
#         satellite_filepath = os.path.join(satellite_dir, satellite_filename)

#         # Find matching mask by base name
#         base_name = '_'.join(satellite_filename.split('_')[:2])
#         mask_filename = next((f for f in os.listdir(mask_dir)
#                               if base_name in f and f.endswith('.png')), None)

#         if mask_filename:
#             mask_filepath = os.path.join(mask_dir, mask_filename)

#             # Check if files are already processed
#             tile_filename = f"{os.path.splitext(satellite_filename)[0]}_tile_0_0.tif"
#             tile_filepath = os.path.join(output_satellite_dir, tile_filename)
#             if not os.path.exists(tile_filepath):
#                 print(f"Processing {satellite_filename} and {mask_filename}...")
#                 cut_images_into_pieces(satellite_filepath, mask_filepath, output_satellite_dir, output_mask_dir, tile_size)
#             else:
#                 print(f"Skipping {satellite_filename} as it has already been cut into pieces.")
#         else:
#             print(f"No matching mask found for {satellite_filename}")

import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import numpy as np
from PIL import Image

# Input directories for satellite images and corresponding masks
satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles'
mask_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled_invalidtonegative"

# Output directories for cut pieces
output_satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/images_cuts_S2'
output_mask_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_cuts_S2'

# Create output directories if they don't exist
os.makedirs(output_satellite_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Tile size
tile_size = 256

def cut_images_into_pieces(satellite_path, mask_path, satellite_output_dir, mask_output_dir, tile_size, check_black_threshold=0.05):
    """
    Cuts a satellite image and mask into smaller tiles and saves them if they meet the threshold for black pixels.
    """
    with rasterio.open(satellite_path) as src_sat:
        with Image.open(mask_path) as mask_img:
            mask_array = np.array(mask_img.convert('L'))  # Convert mask to grayscale

        assert src_sat.width == mask_array.shape[1] and src_sat.height == mask_array.shape[0], \
            f"Mismatch in dimensions for {satellite_path} and {mask_path}"

        meta_sat = src_sat.meta.copy()

        # Calculate the number of tiles in each dimension
        n_tiles_x = src_sat.width // tile_size
        n_tiles_y = src_sat.height // tile_size

        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                # Calculate the window for the current tile
                window = Window(i * tile_size, j * tile_size, tile_size, tile_size)
                sat_tile = src_sat.read(window=window)
                mask_tile = mask_array[j * tile_size:(j + 1) * tile_size, i * tile_size:(i + 1) * tile_size]

                # Check the percentage of black pixels in the mask
                black_pixels = (mask_tile == 5).sum()
                total_pixels = mask_tile.size
                black_fraction = black_pixels / total_pixels

                if black_fraction > check_black_threshold:
                    print(f"Skipping tile ({i}, {j}) due to invalid pixels ({black_fraction:.2%}) in the mask.")
                    continue

                # Update metadata for the tiles
                meta_sat.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': window_transform(window, src_sat.transform)
                })

                # Save the satellite tile
                sat_tile_path = os.path.join(satellite_output_dir, f"{os.path.splitext(os.path.basename(satellite_path))[0]}_tile_{i}_{j}.tif")
                with rasterio.open(sat_tile_path, 'w', **meta_sat) as dst_sat:
                    dst_sat.write(sat_tile)

                # Save the mask tile
                mask_tile_path = os.path.join(output_mask_dir, f"{os.path.splitext(os.path.basename(mask_path))[0]}_tile_{i}_{j}.png")
                mask_tile_image = Image.fromarray(mask_tile)
                mask_tile_image.save(mask_tile_path)

# Process files in the directories
for satellite_filename in os.listdir(satellite_dir):
    if satellite_filename.endswith('.tif'):
        satellite_filepath = os.path.join(satellite_dir, satellite_filename)

        # Extract the tile identifier (e.g., "Biesbosch_tile_1")
        satellite_base_name = os.path.splitext(satellite_filename)[0]
        satellite_tile_id = '_'.join(satellite_base_name.split('_')[-2:])  # Extract the unique tile identifier
        
        # Find the exact matching mask by tile ID
        matching_mask = next(
            (f for f in os.listdir(mask_dir)
             if f.endswith('.png') and '_'.join(f.split('_')[-2:]).split('.')[0] == satellite_tile_id),
            None
        )

        if matching_mask:
            mask_filepath = os.path.join(mask_dir, matching_mask)

            # Check if files are already processed
            tile_filename = f"{os.path.splitext(satellite_filename)[0]}_tile_0_0.tif"
            tile_filepath = os.path.join(output_satellite_dir, tile_filename)
            if not os.path.exists(tile_filepath):
                print(f"Processing {satellite_filename} and {matching_mask}...")
                cut_images_into_pieces(satellite_filepath, mask_filepath, output_satellite_dir, output_mask_dir, tile_size)
            else:
                print(f"Skipping {satellite_filename} as it has already been cut into pieces.")
        else:
            print(f"No matching mask found for {satellite_filename}")


num_satellite_tiles = len(os.listdir(output_satellite_dir))
num_mask_tiles = len(os.listdir(output_mask_dir))
print(f"Number of satellite tiles: {num_satellite_tiles}")
print(f"Number of mask tiles: {num_mask_tiles}")