import os
import shutil
import random

# Input directories
images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images'
masks_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks'

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

# Shuffle the filenames
combined = list(zip(image_filenames, mask_filenames))
random.shuffle(combined)
image_filenames[:], mask_filenames[:] = zip(*combined)

# Split into training and validation sets (90% training, 10% validation)
split_index = int(0.9 * len(image_filenames))
train_image_filenames = image_filenames[:split_index]
val_image_filenames = image_filenames[split_index:]
train_mask_filenames = mask_filenames[:split_index]
val_mask_filenames = mask_filenames[split_index:]

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

print("Files have been split into training and validation sets, and the input directories have been removed.")