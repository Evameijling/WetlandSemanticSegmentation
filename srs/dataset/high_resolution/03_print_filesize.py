# import os
# import glob
# import rasterio
# from PIL import Image

# def check_tif_png_dimensions(tif_dir, png_dir):
#     """
#     For each .tif file in `tif_dir`, look for a .png file in `png_dir`
#     with the same basename. If found, compare and print their dimensions.
#     """

#     # Find all TIF files in the tif_dir
#     tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
#     if not tif_files:
#         print(f"No .tif files found in {tif_dir}")
#         return

#     # Process each TIF
#     for tif_path in tif_files:
#         # Extract the file's base name without extension
#         base_name = os.path.splitext(os.path.basename(tif_path))[0]
        
#         # Build the expected PNG path
#         png_path = os.path.join(png_dir, f"{base_name}.png")
        
#         if not os.path.exists(png_path):
#             # If no corresponding PNG, skip or print a warning
#             print(f"[WARNING] No matching .png found for {tif_path}")
#             continue
        
#         # Open the TIF to read its dimensions
#         with rasterio.open(tif_path) as tif_dataset:
#             tif_height = tif_dataset.height
#             tif_width  = tif_dataset.width
        
#         # Open the PNG (with Pillow) to read its dimensions
#         with Image.open(png_path) as img:
#             png_width, png_height = img.size  # Note that PIL uses (width, height)
        
#         print("------------------------------------------------")
#         print(f"TIF:  {tif_path}")
#         print(f"     -> Width: {tif_width}, Height: {tif_height}")
#         print(f"PNG:  {png_path}")
#         print(f"     -> Width: {png_width}, Height: {png_height}")
        
#         # Check if dimensions match
#         if (tif_width == png_width) and (tif_height == png_height):
#             print(f"[OK] Dimensions match for {base_name}")
#         else:
#             print(f"[MISMATCH] Dimensions differ for {base_name}")

# if __name__ == "__main__":
#     # Example usage:
#     tif_input_dir = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles"
#     png_input_dir = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled"
    
#     check_tif_png_dimensions(tif_input_dir, png_input_dir)

import os
import glob
import rasterio
from PIL import Image

def check_tif_png_dimensions(tif_dir, png_dir):
    """
    For each .tif file in `tif_dir`, look for a .png file in `png_dir`
    where the PNG's basename is a substring of the TIF's basename.
    If found, compare and print their dimensions.
    """
    # Find all TIF files in the tif_dir
    tif_files = glob.glob(os.path.join(tif_dir, '*.tif'))
    if not tif_files:
        print(f"No .tif files found in {tif_dir}")
        return

    # Find all PNG files in the png_dir
    png_files = glob.glob(os.path.join(png_dir, '*.png'))
    if not png_files:
        print(f"No .png files found in {png_dir}")
        return

    # Process each TIF
    for tif_path in tif_files:
        # Extract the TIF's base name without extension
        tif_base_name = os.path.splitext(os.path.basename(tif_path))[0]

        # Find a matching PNG file (substring match)
        matching_png = None
        for png_path in png_files:
            png_base_name = os.path.splitext(os.path.basename(png_path))[0]
            if png_base_name in tif_base_name:
                matching_png = png_path
                break
        
        if not matching_png:
            # If no matching PNG found, print a warning and skip
            print(f"[WARNING] No matching .png found for {tif_path}")
            continue

        # Open the TIF to read its dimensions
        with rasterio.open(tif_path) as tif_dataset:
            tif_height = tif_dataset.height
            tif_width = tif_dataset.width

        # Open the PNG to read its dimensions
        with Image.open(matching_png) as img:
            png_width, png_height = img.size  # PIL uses (width, height)

        print("------------------------------------------------")
        print(f"TIF:  {tif_path}")
        print(f"     -> Width: {tif_width}, Height: {tif_height}")
        print(f"PNG:  {matching_png}")
        print(f"     -> Width: {png_width}, Height: {png_height}")

        # Check if dimensions match
        if (tif_width == png_width) and (tif_height == png_height):
            print(f"[OK] Dimensions match for {tif_base_name}")
        else:
            print(f"[MISMATCH] Dimensions differ for {tif_base_name}")

if __name__ == "__main__":
    # Example usage:
    tif_input_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles"
    png_input_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled"
    
    check_tif_png_dimensions(tif_input_dir, png_input_dir)
