import os
import rasterio
import numpy as np

# Input and output directories
input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/imagery_tiles/annotation_images'
output_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/imagery_tiles/annotation_images_selectedbands'

# Band Index Overview Pleiades NEO
# 1: B1 - Red   
# 2: B2 - Green
# 3: B3 - Blue
# 4: B4 - Near Infrared
# 5: B5 - Red Edge 1
# 6: B6 - Deep Blue

# Band Index Overview SuperView NEO-1
# 1: B1 - Red
# 2: B2 - Blue
# 3: B3 - Green
# 4: B4 - Near Infrared

# Extract only the red, blue, green, and near infrared bands
bands_to_extract = [1, 2, 3, 4]

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all TIFF files in the input directory
for input_filename in os.listdir(input_dir):
    if input_filename.endswith('.tif'):
        # Full path to the input file
        input_filepath = os.path.join(input_dir, input_filename)
        
        # Construct the output file path
        output_filepath = os.path.join(output_dir, input_filename)
        
        # Check if the file already exists in the output directory
        if os.path.exists(output_filepath):
            print(f"File already exists, skipping: {output_filepath}")
            continue
        
        try:
            # Open the input file
            with rasterio.open(input_filepath) as src:
                # Read the specified bands
                band_data = []
                for band in bands_to_extract:
                    band_data.append(src.read(band))
                
                # Stack the bands into a single array
                band_data = np.stack(band_data)
                
                # Update metadata for the output file
                meta = src.meta.copy()
                meta.update({
                    'count': len(bands_to_extract),
                    'dtype': band_data.dtype
                })
                
                # Write the selected bands to the output file
                with rasterio.open(output_filepath, 'w', **meta) as dst:
                    dst.write(band_data)
                
                print(f"Processed {input_filename} and saved to {output_filepath}")
        
        except rasterio.errors.RasterioIOError as e:
            print(f"Error processing {input_filename}: {e}")