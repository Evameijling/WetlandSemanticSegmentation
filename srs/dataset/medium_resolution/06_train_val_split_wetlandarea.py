import os
import shutil
import re

# Input directories
images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_secondbatch'
masks_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_secondbatch'

# Output directories
images_train_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_train'
images_val_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val'
masks_train_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_train'
masks_val_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_val'

# Ensure output directories exist
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(masks_train_dir, exist_ok=True)
os.makedirs(masks_val_dir, exist_ok=True)

# Get sorted list of image and mask filenames
image_filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
mask_filenames = sorted([f for f in os.listdir(masks_dir) if f.endswith('.tif')])

# Extract base names without extensions and prefixes
image_basenames = {os.path.splitext(f)[0].split('Sentinel2')[1] for f in image_filenames}
mask_basenames = {os.path.splitext(f)[0].split('DynamicWorld')[1] for f in mask_filenames}

# Find the intersection of the base names
common_basenames = image_basenames.intersection(mask_basenames)

# Filter filenames based on the common base names
image_filenames = [f for f in image_filenames if os.path.splitext(f)[0].split('Sentinel2')[1] in common_basenames]
mask_filenames = [f for f in mask_filenames if os.path.splitext(f)[0].split('DynamicWorld')[1] in common_basenames]

# Ensure the filenames match
print(f"Number of images: {len(image_filenames)}")
print(f"Number of masks: {len(mask_filenames)}")
assert len(image_filenames) == len(mask_filenames), "Mismatch between number of images and masks after filtering"

# Function to check if the filename contains "biesbosch"
def is_validation_filename(filename):
    return "biesbosch" in filename.lower()

# Split filenames into training and validation sets based on filename
train_image_filenames = []
val_image_filenames = []
train_mask_filenames = []
val_mask_filenames = []

for img_filename, mask_filename in zip(image_filenames, mask_filenames):
    if is_validation_filename(img_filename):
        val_image_filenames.append(img_filename)
        val_mask_filenames.append(mask_filename)
    else:
        train_image_filenames.append(img_filename)
        train_mask_filenames.append(mask_filename)

# Move files to the appropriate directories
for filename in train_image_filenames:
    shutil.move(os.path.join(images_dir, filename), os.path.join(images_train_dir, filename))

for filename in val_image_filenames:
    shutil.move(os.path.join(images_dir, filename), os.path.join(images_val_dir, filename))

for filename in train_mask_filenames:
    shutil.move(os.path.join(masks_dir, filename), os.path.join(masks_train_dir, filename))

for filename in val_mask_filenames:
    shutil.move(os.path.join(masks_dir, filename), os.path.join(masks_val_dir, filename))

# Remove the input directories after the split
shutil.rmtree(images_dir)
shutil.rmtree(masks_dir)

print("Files have been split into training and validation sets based on filename, and the input directories have been removed.")

