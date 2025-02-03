import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os

# Path to your input TIFF file
tif_path = '/home/egmelich/SatelliteMAE/tif_evaluation/dw_-62.4931115326_5.7583918135-20181215.tif'

# Output path for the PNG image
output_png_path = os.path.join(os.path.dirname(tif_path), 'output_image.png')

# Open the TIFF file
with rasterio.open(tif_path) as dataset:
    # Read the first band of the raster (assuming single band)
    band = dataset.read(1)

    # Get unique pixel values (classes) in the band
    unique_values = np.unique(band)
    print("Unique pixel values:", unique_values)

    # Assign a unique colour to each unique value
    cmap = plt.cm.get_cmap('tab20', len(unique_values))  # Choose a colormap with enough colours
    color_dict = {value: cmap(i) for i, value in enumerate(unique_values)}

    # Create a colour-mapped array for visualisation
    color_mapped = np.zeros((band.shape[0], band.shape[1], 3), dtype=np.uint8)
    for value, color in color_dict.items():
        mask = band == value
        color_mapped[mask] = np.array(color[:3]) * 255  # Convert from 0-1 to 0-255 range for RGB

    # Display the colour-mapped image
    plt.imshow(color_mapped)
    plt.axis('off')
    plt.title('Pixel Value Classes')

    # Save the colour-mapped image as a PNG
    Image.fromarray(color_mapped).save(output_png_path)
    print(f"Image saved as PNG at {output_png_path}")

plt.show()
