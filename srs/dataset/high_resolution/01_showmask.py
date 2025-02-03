import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------
# 1) Define your class colors and names as lists:
#    Index i in class_colors corresponds to index i in class_names.
# ------------------------------------------------------------------------------
class_colors = [
    (0, 0, 0),       # 0: Negative
    (255, 0, 0),     # 1: Built
    (110, 93, 3),    # 2: Flooded Soil
    (20, 102, 0),    # 3: Forest
    (83, 251, 84),   # 4: Grass & Farmland
    (255, 0, 204),   # 5: Invalid Pixels
    (224, 255, 6),   # 6: Reed & Rough
    (255, 214, 0),   # 7: Shrubs
    (0, 195, 206),   # 8: Water
]

class_names = [
    "Negative",
    "Built",
    "Flooded Soil",
    "Forest",
    "Grass & Farmland",
    "Invalid Pixels",
    "Reed & Rough",
    "Shrubs",
    "Water"
]

# ------------------------------------------------------------------------------
# 2) Define the input and output directories
# ------------------------------------------------------------------------------
input_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks/"
output_dir = "/home/egmelich/SatelliteMAE/Preprocessing_Satellietdataportaal_Biesbosch/masks_highres/no_legend/"
os.makedirs(output_dir, exist_ok=True)

def load_mask_and_unique_values(png_path):
    """
    Loads a grayscale mask (single-channel, with values 0..8) and returns 
    the pixel array and its unique values.
    """
    with Image.open(png_path) as img:
        # Make sure image is grayscale
        img = img.convert("L")
        pixel_array = np.array(img)

    unique_pixels = np.unique(pixel_array)
    return pixel_array, unique_pixels

def visualize_mask_and_save(pixel_array, unique_values, output_path):
    """
    Given a grayscale mask (pixel_array) with class indices 0..8, create an 
    RGB color-mapped visualization and save it with a legend.
    """
    # Shape: (height, width)
    height, width = pixel_array.shape

    # Convert lists to NumPy array so we can do "fancy indexing"
    color_array = np.array(class_colors, dtype=np.uint8)  # shape: (9, 3)
    
    # 1) Create an RGB image by indexing into color_array with pixel_array
    #    pixel_array has shape (height, width), so we'll get (height, width, 3)
    color_mapped = color_array[pixel_array]

    # 2) Plot the color-mapped image
    fig, ax = plt.subplots()
    ax.imshow(color_mapped)
    ax.axis('off')  # Remove axis ticks

    # # 3) Build legend items only for classes found in unique_values
    # legend_elements = []
    # for val in unique_values:
    #     # val is in [0..8], so we can directly use class_colors[val]
    #     rgb = np.array(class_colors[val]) / 255.0
    #     label = class_names[val] if val < len(class_names) else f"Class {val}"
    #     legend_elements.append(
    #         Patch(facecolor=rgb, edgecolor='white', label=label)
    #     )

    # if legend_elements:
    #     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # 4) Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[INFO] Saved visualization to: {output_path}")

if __name__ == "__main__":
    # Iterate over all PNG files in the input directory
    for input_png in os.listdir(input_dir):
        if input_png.endswith(".png"):
            input_path = os.path.join(input_dir, input_png)
            
            # 1) Load the mask and get the unique values
            pixel_array, unique_values = load_mask_and_unique_values(input_path)
            print(f"[INFO] Unique class indices in {input_path}: {unique_values}")

            # 2) Construct an output path in your desired output directory
            output_path = os.path.join(output_dir, input_png)

            # 3) Generate and save the color-mapped visualization
            visualize_mask_and_save(pixel_array, unique_values, output_path)
