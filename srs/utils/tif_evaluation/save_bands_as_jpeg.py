import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from matplotlib import cm

def save_bands_with_colors(
    input_tiff_path,
    output_folder,
    format='JPEG',
    quality=95,
    downsample_factor=1
):
    """
    Save each band of a GeoTIFF file as a separate JPEG image with corresponding colors.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Define corresponding colormaps for each band
    band_colormaps = {
        1: cm.plasma,
        2: cm.Blues,
        3: cm.Greens,
        4: cm.Reds,
        5: cm.cividis,
        6: cm.magma,
        7: cm.viridis,
        8: cm.inferno,
        9: cm.cool,
        10: cm.winter,
        11: cm.summer,
        12: cm.spring,
        13: cm.autumn,
        14: cm.bone,
        15: cm.hot,
        16: cm.copper,
        17: cm.gray,
        18: cm.Purples,
        19: cm.viridis,
        20: cm.coolwarm,
        21: cm.jet,
        22: cm.hsv,
        23: cm.terrain
    }

    print(f"Opening {input_tiff_path} with Rasterio ...")
    with rasterio.open(input_tiff_path) as src:
        band_count = src.count

        new_height = src.height // downsample_factor
        new_width = src.width // downsample_factor

        if downsample_factor > 1:
            print(f"Downsampling from {(src.height, src.width)} to {(new_height, new_width)}")

        for band_idx in range(1, band_count + 1):
            print(f"Processing Band {band_idx} ...")

            # Read the band with optional downsampling
            band_data = src.read(
                band_idx,
                out_shape=(new_height, new_width),
                resampling=Resampling.lanczos
            )

            # Convert to float for processing
            band_data = band_data.astype(np.float32)

            # Detect invalid (zero) pixels and create a mask
            invalid_mask = band_data == 0

            # Apply 2–98 percentile stretch for valid pixels only
            valid_mask = ~invalid_mask

            if np.any(valid_mask):
                p2 = np.percentile(band_data[valid_mask], 2)
                p98 = np.percentile(band_data[valid_mask], 98)

                # Clip to the 2%–98% range
                band_data_clipped = np.clip(band_data, p2, p98)

                # Scale to [0..1] for colormap application
                if p98 > p2:  # Avoid division by zero if p2 == p98
                    band_data_scaled = (band_data_clipped - p2) / (p98 - p2)
                else:
                    band_data_scaled = band_data_clipped  # Fallback if everything is the same
            else:
                band_data_scaled = band_data  # If all pixels are invalid, keep as is

            # Apply colormap to valid pixels
            colormap = band_colormaps.get(band_idx, cm.viridis)  # Default to viridis if not defined
            band_colored = (colormap(band_data_scaled)[:, :, :3] * 255).astype(np.uint8)  # Convert RGBA to RGB

            # Apply the invalid mask to keep invalid pixels black
            band_colored[invalid_mask] = [0, 0, 0]

            # Create a Pillow image for the single band
            img = Image.fromarray(band_colored, mode='RGB')

            # Construct the output file path
            output_file_path = os.path.join(output_folder, f"Band_{band_idx}.{format.lower()}")

            # Save the band as a JPEG/PNG
            if format.upper() == 'JPEG':
                print(f"Saving Band {band_idx} as JPEG -> {output_file_path}")
                img.save(output_file_path, format='JPEG', quality=quality, optimize=True)
            else:
                print(f"Saving Band {band_idx} as PNG -> {output_file_path}")
                img.save(output_file_path, format='PNG', optimize=True, compress_level=9)

            print(f"Band {band_idx} saved successfully.")

def main():
    # Example usage
    input_image = 'Biesbosch_Sentinel2_2024-08-12'

    # Set output folder
    output_folder = '/home/egmelich/SatelliteMAE/Preprocessing_Sentinel2/bands_output'

    # Choose desired output format: 'JPEG' or 'PNG'
    output_format = 'JPEG'

    # Construct input file path
    input_tiff_path = f'/projects/0/prjs1235/DynamicWorld_GEEData/original/{input_image}.tif'

    # Other parameters
    jpeg_quality = 95
    down_factor = 1

    save_bands_with_colors(
        input_tiff_path=input_tiff_path,
        output_folder=output_folder,
        format=output_format,
        quality=jpeg_quality,
        downsample_factor=down_factor
    )

if __name__ == '__main__':
    main()