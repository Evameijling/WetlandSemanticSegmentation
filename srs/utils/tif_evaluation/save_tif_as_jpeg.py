############# CONVERT SINGLE TIF TO JPG ############# 

# import os
# import numpy as np
# import rasterio
# from rasterio.enums import Resampling
# from PIL import Image

# def compress_tiff_to_jpeg_or_png(
#     input_tiff_path,
#     output_file_path,
#     format='JPEG',
#     quality=95,
#     downsample_factor=1,
#     max_size_bytes=2 * 1024**3
# ):
#     """
#     Convert a single large GeoTIFF file to JPEG/PNG, applying a 2–98 percentile stretch
#     on each of the first three bands (R, G, B) before converting to 8-bit.
#     """
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

#     print(f"Opening {input_tiff_path} with Rasterio ...")
#     with rasterio.open(input_tiff_path) as src:
#         # Select ONLY the first three bands, assuming they are R, G, B
#         bands_to_read = [1, 2, 3]

#         new_height = src.height // downsample_factor
#         new_width  = src.width // downsample_factor

#         if downsample_factor > 1:
#             print(f"Downsampling from {(src.height, src.width)} to {(new_height, new_width)}")

#         # Read only the first three bands, with optional downsampling
#         data_3d = src.read(
#             bands_to_read,
#             out_shape=(len(bands_to_read), new_height, new_width),
#             resampling=Resampling.lanczos
#         )
#         # data_3d shape: (3, height, width) in 16-bit (or other)

#     print(f"Data shape (after reading first 3 bands): {data_3d.shape}")

#     # Convert to float for processing
#     data_3d = data_3d.astype(np.float32)

#     # Apply 2–98 percentile stretch per band
#     for b in range(data_3d.shape[0]):
#         band_data = data_3d[b, :, :]

#         # Compute lower (p2) and upper (p98) percentiles
#         p2 = np.percentile(band_data, 2)
#         p98 = np.percentile(band_data, 98)

#         # Clip to the 2%–98% range
#         band_data_clipped = np.clip(band_data, p2, p98)

#         # Scale to [0..255]
#         if p98 > p2:  # Avoid division by zero if p2==p98
#             band_data_scaled = (band_data_clipped - p2) / (p98 - p2) * 255.0
#         else:
#             band_data_scaled = band_data_clipped  # fallback if everything is the same

#         data_3d[b, :, :] = band_data_scaled

#     # Transpose to (height, width, 3) for Pillow
#     data_3d = np.transpose(data_3d, (1, 2, 0)).astype(np.uint8)

#     # Create a Pillow image in RGB mode
#     img = Image.fromarray(data_3d, mode='RGB')

#     if format.upper() == 'JPEG':
#         print(f"Saving as JPEG -> {output_file_path} with quality={quality}")
#         img.save(output_file_path, format='JPEG', quality=quality, optimize=True)
#     else:
#         print(f"Saving as PNG -> {output_file_path}")
#         img.save(output_file_path, format='PNG', optimize=True, compress_level=9)

#     # Check final size
#     final_size = os.path.getsize(output_file_path)
#     print(f"Saved {output_file_path} (size: {final_size / (1024**2):.2f} MB)")

#     if final_size > max_size_bytes:
#         print(f"WARNING: {output_file_path} is still larger than {max_size_bytes} bytes.")
#         print("Consider lowering `quality` or increasing `downsample_factor`.")

# def main():
#     # Example usage
#     input_image = '20240921_104931_PNEO-04_2_1_30cm_RD_12bit_RGBNED_Biesbosch_clipped'

#     # Choose desired output format: 'JPEG' or 'PNG'
#     output_format = 'JPEG'
    
#     # Set the file extension
#     output_ext = 'jpg' if output_format.upper() == 'JPEG' else 'png'

#     # Construct paths
#     input_tiff_path  = f'/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_Biesbosch/{input_image}.tif'
#     output_file_path = f'/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_Biesbosch_jpg/{input_image}.{output_ext}'

#     # Other parameters
#     jpeg_quality = 85
#     down_factor = 1
#     max_size = 2 * 1024**3  # 2 GB

#     compress_tiff_to_jpeg_or_png(
#         input_tiff_path=input_tiff_path,
#         output_file_path=output_file_path,
#         format=output_format,
#         quality=jpeg_quality,
#         downsample_factor=down_factor,
#         max_size_bytes=max_size
#     )

# if __name__ == '__main__':
#     main()


############# CONVERT FOLDER OF TIFS TO FOLDER OF JPGS ############# 

import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image

def compress_tiff_to_jpeg_or_png(
    input_tiff_path,
    output_file_path,
    format='JPEG',
    quality=95,
    downsample_factor=1,
    max_size_bytes=2 * 1024**3
):
    """
    Convert a single large GeoTIFF file to JPEG/PNG, applying a 2–98 percentile stretch
    on each of the first three bands (R, G, B) before converting to 8-bit.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    print(f"\nOpening {input_tiff_path} with Rasterio ...")
    with rasterio.open(input_tiff_path) as src:
        # Select ONLY the first three bands, assuming they are R, G, B
        bands_to_read = [3, 2, 1]

        new_height = src.height // downsample_factor
        new_width  = src.width // downsample_factor

        if downsample_factor > 1:
            print(f"Downsampling from {(src.height, src.width)} to {(new_height, new_width)}")

        # Read only the first three bands, with optional downsampling
        data_3d = src.read(
            bands_to_read,
            out_shape=(len(bands_to_read), new_height, new_width),
            resampling=Resampling.lanczos
        )
        # data_3d shape: (3, height, width) in 16-bit (or other)

    print(f"Data shape (after reading first 3 bands): {data_3d.shape}")

    # Convert to float for processing
    data_3d = data_3d.astype(np.float32)

    # Apply 2–98 percentile stretch per band
    for b in range(data_3d.shape[0]):
        band_data = data_3d[b, :, :]

        # Compute lower (p2) and upper (p98) percentiles
        p2 = np.percentile(band_data, 2)
        p98 = np.percentile(band_data, 98)

        # Clip to the 2%–98% range
        band_data_clipped = np.clip(band_data, p2, p98)

        # Scale to [0..255]
        if p98 > p2:  # Avoid division by zero if p2 == p98
            band_data_scaled = (band_data_clipped - p2) / (p98 - p2) * 255.0
        else:
            band_data_scaled = band_data_clipped  # fallback if everything is the same

        data_3d[b, :, :] = band_data_scaled

    # Transpose to (height, width, 3) for Pillow
    data_3d = np.transpose(data_3d, (1, 2, 0)).astype(np.uint8)

    # Create a Pillow image in RGB mode
    img = Image.fromarray(data_3d, mode='RGB')

    if format.upper() == 'JPEG':
        print(f"Saving as JPEG -> {output_file_path} with quality={quality}")
        img.save(output_file_path, format='JPEG', quality=quality, optimize=True)
    else:
        print(f"Saving as PNG -> {output_file_path}")
        img.save(output_file_path, format='PNG', optimize=True, compress_level=9)

    # Check final size
    final_size = os.path.getsize(output_file_path)
    print(f"Saved {output_file_path} (size: {final_size / (1024**2):.2f} MB)")

    if final_size > max_size_bytes:
        print(f"WARNING: {output_file_path} is still larger than {max_size_bytes} bytes.")
        print("Consider lowering `quality` or increasing `downsample_factor`.")

def main():
    # Folder containing input TIFFs
    input_folder = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles'
    # Folder to save output images
    output_folder = '/home/egmelich/SatelliteMAE/Preprocessing_Satellietdataportaal_Biesbosch/S2imagery_tiles'
    
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Choose desired output format: 'JPEG' or 'PNG'
    output_format = 'JPEG'
    
    # Derive extension based on format
    output_ext = 'jpg' if output_format.upper() == 'JPEG' else 'png'

    # Other parameters
    jpeg_quality = 95
    down_factor = 1
    max_size = 2 * 1024**3  # 2 GB

    # Get a sorted list of all .tif files in the input folder
    tiff_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))

    print(f"Found {len(tiff_files)} TIFF files in {input_folder}.")

    for idx, tif_path in enumerate(tiff_files, start=1):
        # Get just the filename without extension
        base_name = os.path.splitext(os.path.basename(tif_path))[0]

        # Construct output filename
        output_file_path = os.path.join(
            output_folder,
            f"{base_name}.{output_ext}"
        )

        print(f"\nProcessing file {idx}/{len(tiff_files)}: {os.path.basename(tif_path)}")
        
        # Call the compression function
        compress_tiff_to_jpeg_or_png(
            input_tiff_path=tif_path,
            output_file_path=output_file_path,
            format=output_format,
            quality=jpeg_quality,
            downsample_factor=down_factor,
            max_size_bytes=max_size
        )

        print(f"Finished processing: {output_file_path}")

    print("\nAll files processed.")

if __name__ == '__main__':
    main()
