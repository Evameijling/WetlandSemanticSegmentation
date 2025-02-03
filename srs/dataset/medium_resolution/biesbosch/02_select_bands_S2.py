import os
import rasterio
import numpy as np

# Input and output directories
input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles'
output_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles/selectedbands'

# Bands to extract (B2, B3, B4, B5, B6, B7, B8, B11, B12)
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands

# Band Index Overview
# 1  B1  = Aerosols
# 2  B2  = Blue
# 3  B3  = Green
# 4  B4  = Red
# 5  B5  = Red Edge 1
# 6  B6  = Red Edge 2
# 7  B7  = Red Edge 3
# 8  B8  = NIR
# 9  B8A = Red Edge 4
# 10 B9  = Water vapor
# 11 B11 = SWIR 1
# 12 B12 = SWIR 2
# 13 AOT = Aerosol Optical Thickness
# 14 WVP = Water Vapor Pressure
# 15 SCL = Scene Classification Map
# 16 TCI_R = True Color Image, Red channel
# 17 TCI_G = True Color Image, Green channel
# 18 TCI_B = True Color Image, Blue channel
# 19 MSK_CLDPRB = Cloud Probability Map
# 20 MSK_SNWPRB = Snow Probability Map
# 21 QA10 = Always empty
# 22 QA20 = Always empty
# 23 QA60 = Cloud mask

# check how many bands the input image has
with rasterio.open(os.path.join(input_dir, os.listdir(input_dir)[0])) as src:
    print('number of bands:', src.count)

# print resolution of the input image
with rasterio.open(os.path.join(input_dir, os.listdir(input_dir)[0])) as src:
    print('resolution:', src.res)

# Extract only the red, blue, green, and near infrared bands
bands_to_extract = [2, 3, 4, 8]

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