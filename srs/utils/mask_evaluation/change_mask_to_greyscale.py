import os
import sys
import numpy as np
from PIL import Image

def map_grayscale_values(image):
    """
    Maps unique grayscale pixel values in the image to a predefined mapping.

    :param image: Grayscale PIL.Image object
    :return: PIL.Image object with remapped pixel values
    """
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Print the unique grayscale values in the image
    unique_values = np.unique(img_array)
    print(f"[INFO] Unique grayscale values before remapping: {unique_values}")

    # Predefined mapping for grayscale values
    value_map = {
        0: 0,
        66: 1,
        76: 2,
        99: 3,
        138: 4,
        182: 5,
        202: 6,
        217: 7,
    }

    # Apply the predefined mapping to the image array, defaulting to 0 for unmapped values
    mapped_array = np.vectorize(value_map.get, otypes=[np.uint8])(img_array)

    # Convert back to a PIL.Image object
    mapped_image = Image.fromarray(mapped_array)

    return mapped_image

def overwrite_pngs_in_directory(input_directory):
    """
    Converts all .png files in the specified directory to grayscale, remaps unique pixel values,
    and overwrites the original files with their remapped grayscale versions.

    :param input_directory: Path to the directory containing .png files
    """
    if not os.path.isdir(input_directory):
        print(f"[ERROR] Directory does not exist: {input_directory}")
        sys.exit(1)

    # List all .png files in the directory
    png_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.png')]

    if not png_files:
        print(f"[INFO] No .png files found in directory: {input_directory}")
        return

    for png_file in png_files:
        input_png_path = os.path.join(input_directory, png_file)
        try:
            with Image.open(input_png_path) as img:
                # Convert to single-channel grayscale
                img_gray = img.convert('L')

                # Map the unique grayscale values
                img_mapped = map_grayscale_values(img_gray)

            # Overwrite the original .png file with the remapped grayscale version
            img_mapped.save(input_png_path)
            print(f"[INFO] Overwrote {png_file} with remapped grayscale version.")

            # Print the unique grayscale values after remapping
            unique_values_after = np.unique(np.array(img_mapped))
            print(f"[INFO] Unique grayscale values after remapping: {unique_values_after}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {png_file}: {e}")

if __name__ == "__main__":
    """
    Define the directory to process here.
    """
    input_directory_path = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks"
    overwrite_pngs_in_directory(input_directory_path)
