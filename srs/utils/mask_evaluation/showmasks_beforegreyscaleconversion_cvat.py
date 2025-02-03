import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1) Define your color/label dictionaries here:
color_mapping = {
    0:   (0, 0, 0),      
    66:  (20, 102, 0),      # Forrest   
    76:  (255, 0, 0),       # Built
    138: (0, 195, 206),     # Water   
    182: (83, 251, 84),     # Grass & Farmland
    202: (255, 214, 0),     # Shrubs  
    217: (224, 255, 6),     # Reed & Rough  
    
}

labels = {
    0:   "Negative",
    66:  "Forest",
    76:  "Built",
    138: "Water",
    182: "Grass & Farmland",
    202: "Shrubs",
    217: "Reed & Rough",

}

# 2) Define the output directory
output_dir = "/home/egmelich/SatelliteMAE/Preprocessing_Satellietdataportaal_Biesbosch/SDPmasks"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def get_unique_pixels_as_grayscale(png_path):
    """
    Loads a PNG (possibly multi-channel), forces it to grayscale,
    and returns both the 2D array and the unique grayscale values.
    """
    with Image.open(png_path) as img:
        # Convert to single-channel grayscale
        img_gray = img.convert('L')
        pixel_array = np.array(img_gray)  # shape (H, W)

    # Identify unique pixel values
    unique_pixels = np.unique(pixel_array)
    return pixel_array, unique_pixels

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
    input_png = "/projects/0/prjs1235/Satellietdataportaal_data/masks_manualannotation/20230930_103458_Made_tile_10.png"

    # 1) Read the mask as grayscale, get the unique values
    pixel_array, unique_values = get_unique_pixels_as_grayscale(input_png)

    # 2) Print the unique grayscale values
    print(f"[INFO] Unique grayscale values in {input_png}: {unique_values}")

    # 3) Construct an output path in your desired output directory
    base_name = os.path.basename(input_png)
    output_png = os.path.join(
        output_dir,
    )

    # 4) Generate and save the color-mapped visualization
    visualize_mask_and_save(pixel_array, unique_values, output_png)
