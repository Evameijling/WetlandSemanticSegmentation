import os
import rasterio
from rasterio.windows import Window

# Input directories containing the images and masks
input_dirs = [
    '/projects/0/prjs1235/DynamicWorld_GEEData/images',
    '/projects/0/prjs1235/DynamicWorld_GEEData/masks'
]

# # Set resolution (in meters per pixel)
# resolution = 10  # 10 meters per pixel

# # Desired image size in meters (e.g., 1km x 1km)
# desired_size_meters = 1000  # 1000 meters (1 km)

# # Calculate tile dimensions in pixels
# tile_size_x = int(desired_size_meters / resolution)
# tile_size_y = int(desired_size_meters / resolution)

# Desired image size in pixels
tile_size_x = 256  # 256 pixels wide
tile_size_y = 256  # 256 pixels tall

# Function to check for black pixels in an image
def has_too_many_black_pixels(image_path, threshold=0.1):
    with rasterio.open(image_path) as dataset:
        data = dataset.read()
        num_black_pixels = (data == 0).sum()
        total_pixels = data.size
        return (num_black_pixels / total_pixels) > threshold

# Loop over all input directories
for in_dir in input_dirs:
    # Loop over all TIFF files in the input directory
    for input_filename in os.listdir(in_dir):
        if input_filename.endswith('.tif'):
            # Full path to the input file
            input_filepath = os.path.join(in_dir, input_filename)
            
            # Create an output directory for the current TIFF file
            base_name = os.path.splitext(input_filename)[0]

            # Open the dataset
            with rasterio.open(input_filepath) as dataset:
                xsize, ysize = dataset.width, dataset.height

                # Loop over the raster and create tiles
                for i in range(tile_size_x, xsize - tile_size_x, tile_size_x):
                    for j in range(tile_size_y, ysize - tile_size_y, tile_size_y):
                        # Define window for current tile
                        window = Window(i, j, tile_size_x, tile_size_y)
                        
                        # Read data from the window
                        tile_data = dataset.read(window=window)

                        # Construct output filename for the tile
                        output_tile = os.path.join(in_dir, f"{base_name}_tile_{i}_{j}.tif")
                        
                        # Define metadata for the tile
                        tile_transform = dataset.window_transform(window)
                        tile_meta = dataset.meta.copy()
                        tile_meta.update({
                            "driver": "GTiff",
                            "height": tile_data.shape[1],
                            "width": tile_data.shape[2],
                            "transform": tile_transform
                        })
                        
                        # Write the tile to a new file
                        with rasterio.open(output_tile, 'w', **tile_meta) as dest:
                            dest.write(tile_data)

            # Remove the original file after processing
            os.remove(input_filepath)
            print(f"Processed {input_filename}, saved tiles in the same directory, and removed the original file.")

# Check for black pixels in the generated tiles that contain "Sentinel2" in their filenames
for tile_filename in os.listdir(in_dir):
    print(f"Checking tile {tile_filename} for black pixels...")
    if tile_filename.endswith('.tif') and "Sentinel2" in tile_filename:
        tile_filepath = os.path.join(in_dir, tile_filename)
        if has_too_many_black_pixels(tile_filepath):
            print(f"Tile {tile_filename} contains more than 10% black pixels and will be removed.")
            os.remove(tile_filepath)
            
            # Remove the corresponding DynamicWorld file
            dynamicworld_filename = tile_filename.replace("Sentinel2", "DynamicWorld")
            dynamicworld_filepath = os.path.join(in_dir, dynamicworld_filename)
            if os.path.exists(dynamicworld_filepath):
                os.remove(dynamicworld_filepath)
                print(f"Corresponding DynamicWorld file {dynamicworld_filename} has been removed.")