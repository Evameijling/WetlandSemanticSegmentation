import os
import shutil
from tqdm import tqdm

# Input directory containing the images
input_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/original'

# Output directories for Sentinel-2 and Dynamic World images
sentinel2_output_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images'
dynamicworld_output_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks'

# Ensure the output directories exist
os.makedirs(sentinel2_output_dir, exist_ok=True)
os.makedirs(dynamicworld_output_dir, exist_ok=True)

# Get the list of files in the input directory
files = os.listdir(input_dir)

# Loop over all files in the input directory with a progress bar
for filename in tqdm(files, desc="Processing files"):
    # Full path to the input file
    input_filepath = os.path.join(input_dir, filename)
    
    # Check if the file is a Sentinel-2 image
    if 'Sentinel2' in filename:
        # Copy the file to the Sentinel-2 output directory
        shutil.copy(input_filepath, os.path.join(sentinel2_output_dir, filename))
        # # Move the file to the Sentinel-2 output directory
        # shutil.move(input_filepath, os.path.join(sentinel2_output_dir, filename))
    
    # Check if the file is a Dynamic World image
    elif 'DynamicWorld' in filename:
        # Copy the file to the Dynamic World output directory
        shutil.copy(input_filepath, os.path.join(dynamicworld_output_dir, filename))
        # # Move the file to the Dynamic World output directory
        # shutil.move(input_filepath, os.path.join(dynamicworld_output_dir, filename))

print("File separation complete.")