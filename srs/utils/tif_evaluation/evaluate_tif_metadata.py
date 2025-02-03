import os
import numpy as np
import rasterio

def evaluate_tif_metadata(file_path):
    with rasterio.open(file_path) as dataset:
        crs = dataset.crs
        transform = dataset.transform
        bbox = dataset.bounds
        resolution = dataset.res
        pixel_width, pixel_height = resolution
        num_bands = dataset.count
        meta = dataset.meta

        # Check if the file is georeferenced
        is_georeferenced = crs is not None and transform is not None

        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB

        # Print metadata
        print("File Size: {:.2f} MB".format(file_size))
        print("Metadata:", meta)
        print("CRS:", crs)
        print("Transform:", transform)
        print("Width:", dataset.width)
        print("Height:", dataset.height)
        print("Bounding Box:", bbox)
        print("Resolution (Width):", pixel_width)
        print("Resolution (Height):", pixel_height)
        print("Number of Bands:", num_bands)
        print("Is Georeferenced:", is_georeferenced)

        # Calculate and print min, max, zero-value pixels, and resolution for each band
        for band in range(1, num_bands + 1):  # Bands are 1-indexed in Rasterio
            band_data = dataset.read(band)
            min_val = band_data.min()
            max_val = band_data.max()
            zero_pixel_count = (band_data == 0).sum()  # Count pixels with value 0
            data_type = band_data.dtype  # Data type of the band
            bit_depth = np.iinfo(data_type).bits  # Get bit depth from the data type

            print(f"Band {band}: Min Pixel Value = {min_val}, Max Pixel Value = {max_val}, Zero Pixels = {zero_pixel_count}")
            print(f"Band {band}: Data Type = {data_type}, Bit Depth = {bit_depth}-bit")
            print(f"Band {band}: Resolution (Width) = {pixel_width}, Resolution (Height) = {pixel_height}")

# Example usage
# file_path = '/projects/0/prjs1235/DynamicWorld_GEEData/original_SDPsatimagery/20230415_105231_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Biesbosch_clipped.tif'
# file_path = '/projects/0/prjs1235/DynamicWorld_GEEData/original_SDPsatimagery/20230527_104800_SVNEO-01_30cm_RD_11bit_RGBN_MillingenAanDeRijn_clipped.tif'

# file_path = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val/Biesbosch_Sentinel2_2019-07-30_tile_768_256.tif'

# file_path = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_selectedbands/20240921_104931_PNEO-04_2_1_30cm_RD_12bit_RGBNED_Biesbosch_clipped.tif'
file_path = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmergedmasks/mask_20240921_104931_PNEO-04_2_1_30cm_RD_12bit_RGBNED_Biesbosch_clipped.tif'


evaluate_tif_metadata(file_path)