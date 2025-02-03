import os
import numpy as np
from PIL import Image

# Input and output directories
input_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled"
output_dir = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled_invalidtonegative"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Adjust the extension if needed
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Open the image and process it
        with Image.open(input_path) as img:
            img_array = np.array(img)  # Convert image to numpy array
            
            # Replace pixel value 5 with 0
            img_array[img_array == 5] = 0
            
            # Convert back to an image
            modified_img = Image.fromarray(img_array)
            
            # Save the modified image
            modified_img.save(output_path)

        print(f"Processed and saved: {output_path}")

print("Processing complete!")
