import os
import rasterio
from rasterio.windows import Window

# Paths and filenames
in_path = r'C:\Users\eva.gmelich.meijling\OneDrive - Accenture\Documents\AA satelietdataportaal\tif_files\\'
input_filename = '20230603_104552_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Oostvaardersplassen_clip.tif'
out_path = r'C:\Users\eva.gmelich.meijling\OneDrive - Accenture\Documents\AA satelietdataportaal\tif_tiles\\'
output_filename = 'tile_'

# Tile dimensions in pixels
tile_size_x = 3333
tile_size_y = 3333

# Open the dataset
with rasterio.open(in_path + input_filename) as dataset:
    xsize, ysize = dataset.width, dataset.height
    
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Loop over the raster and create tiles
    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            # Define window for current tile
            window = Window(i, j, tile_size_x, tile_size_y)
            
            # Read data from the window
            tile_data = dataset.read(window=window)
            
            # Check if the tile is a full-size tile or a residual tile
            if tile_data.shape[1] < tile_size_y or tile_data.shape[2] < tile_size_x:
                suffix = '_residual'
            else:
                suffix = ''

            # Construct output filename
            base_name = os.path.splitext(input_filename)[0]  # Get filename without extension
            output_tile = f"{out_path}{output_filename}{i}_{j}{suffix}_{base_name}.tif"
            
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