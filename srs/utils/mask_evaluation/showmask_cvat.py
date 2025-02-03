import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1) Define your color/label dictionaries here:
color_mapping = {
    0:   (0, 0, 0),       # Negative
    1:  (20, 102, 0),     # Forest   
    2:  (255, 0, 0),      # Built
    3: (255, 0, 204),     # Invalid Pixels
    4: (0, 195, 206),     # Water   
    5: (83, 251, 84),     # Grass & Farmland
    6: (255, 214, 0),     # Shrubs  
    7: (224, 255, 6),     # Reed & Rough  
}

labels = {
    0:   "Negative (0)",
    1:   "Forest (1)",
    2:   "Built (2)",
    3:   "Invalid Pixels (3)",
    4:   "Water (4)",
    5:   "Grass & Farmland (5)",
    6:   "Shrubs (6)",
    7:   "Reed & Rough (7)",
}

# 2) Define the output directory
output_dir = "/home/egmelich/SatelliteMAE/Preprocessing_Satellietdataportaal_Biesbosch/SDPmasks"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_mask_and_unique_values(png_path):
    """
    Loads a grayscale mask and returns both the 2D array and its unique values.
    Assumes the input image is already grayscale in sequential range.
    """
    with Image.open(png_path) as img:
        # Load image as grayscale (single channel)
        mask_array = np.array(img)

    # Identify unique pixel values
    unique_pixels = np.unique(mask_array)
    return mask_array, unique_pixels

def visualize_mask_and_save(pixel_array, unique_values, output_path):
    """
    Given a grayscale 2D mask (pixel_array) and its unique values,
    create an RGB color-mapped visualization and save it with a legend.
    """
    height, width = pixel_array.shape
    color_mapped = np.zeros((height, width, 3), dtype=np.uint8)

    # 1) Fill in the RGB colors for each grayscale value
    for val in unique_values:
        if val in color_mapping:
            mask = (pixel_array == val)
            color_mapped[mask] = color_mapping[val]
        else:
            # If there's a pixel value not in color_mapping, skip or set to a default color
            pass

    # 2) Plot the color-mapped image
    fig, ax = plt.subplots()
    ax.imshow(color_mapped)
    ax.axis('off')  # Remove axis ticks/labels

    # 3) Build legend items for the classes we actually found
    legend_elements = []
    for val in unique_values:
        if val in color_mapping:
            rgb = color_mapping[val]
            face_color = np.array(rgb) / 255.0
            class_name = labels[val] if val in labels else f"Class {val}"
            legend_elements.append(
                Patch(facecolor=face_color, edgecolor='white', label=class_name)
            )

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # 4) Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[INFO] Saved visualization to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    input_png = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks/20230430_104233_Sliedrecht_tile_14.png"

    # 1) Load the mask and get the unique values
    pixel_array, unique_values = load_mask_and_unique_values(input_png)

    # 2) Print the unique grayscale values
    print(f"[INFO] Unique grayscale values in {input_png}: {unique_values}")

    # 3) Construct an output path in your desired output directory
    base_name = os.path.basename(input_png)
    output_png = os.path.join(output_dir, base_name)

    # 4) Generate and save the color-mapped visualization
    visualize_mask_and_save(pixel_array, unique_values, output_png)