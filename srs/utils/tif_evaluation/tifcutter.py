import os
import rasterio
from rasterio.windows import Window

filename = "20240921_104931_Biesbosch"

# Paths and filenames
input_filepath = f"/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_Biesbosch/{filename}.tif"
out_dir = f"/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_Biesbosch_tiles/"
output_filename = f"{filename}_tile_"

os.makedirs(out_dir, exist_ok=True)

with rasterio.open(input_filepath) as dataset:
    xsize, ysize = dataset.width, dataset.height
    
    # Calculate tile dimensions for a 4x4 split
    tile_size_x = xsize // 4
    tile_size_y = ysize // 4
    
    tile_number = 1
    # j = rows (top to bottom), i = columns (left to right)
    for j in range(4):
        for i in range(4):
            # Define window for current tile
            window = Window(
                i * tile_size_x,
                j * tile_size_y,
                tile_size_x,
                tile_size_y
            )

            # Read data from the window
            tile_data = dataset.read(window=window)

            # Construct output filename
            output_tile_path = os.path.join(
                out_dir, 
                f"{output_filename}{tile_number}.tif"
            )

            # Define metadata for the tile
            tile_transform = dataset.window_transform(window)
            tile_meta = dataset.meta.copy()
            tile_meta.update({
                "driver": "GTiff",
                "height": tile_data.shape[1],
                "width": tile_data.shape[2],
                "transform": tile_transform
            })

            # Write the tile
            with rasterio.open(output_tile_path, 'w', **tile_meta) as dest:
                dest.write(tile_data)
            
            print(f"Created tile {tile_number} at {output_tile_path}")
            tile_number += 1

print("Finished processing into 16 tiles.")
