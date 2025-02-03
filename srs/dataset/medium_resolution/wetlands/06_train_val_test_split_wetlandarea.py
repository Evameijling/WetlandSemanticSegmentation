import os
import shutil

# Input directories
images_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images'
masks_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks'

# Output directories
images_train_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_train_S2'
images_val_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_val_S2'
images_test_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/images_test_S2'

masks_train_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_train_S2'
masks_val_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_val_S2'
masks_test_dir = '/projects/0/prjs1235/DynamicWorld_GEEData/masks_test_S2'

# Ensure output directories exist
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(images_test_dir, exist_ok=True)

os.makedirs(masks_train_dir, exist_ok=True)
os.makedirs(masks_val_dir, exist_ok=True)
os.makedirs(masks_test_dir, exist_ok=True)

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

# Function to check if the filename contains specific keywords
def is_test_filename(filename):
    return "biesbosch" in filename.lower()

def is_validation_filename(filename):
    return "lauwersmeer" in filename.lower()

# Split filenames into training, validation, and test sets based on filename
train_image_filenames = []
val_image_filenames = []
test_image_filenames = []

train_mask_filenames = []
val_mask_filenames = []
test_mask_filenames = []

for img_filename, mask_filename in zip(image_filenames, mask_filenames):
    if is_test_filename(img_filename):
        test_image_filenames.append(img_filename)
        test_mask_filenames.append(mask_filename)
    elif is_validation_filename(img_filename):
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

for filename in test_image_filenames:
    shutil.move(os.path.join(images_dir, filename), os.path.join(images_test_dir, filename))

for filename in train_mask_filenames:
    shutil.move(os.path.join(masks_dir, filename), os.path.join(masks_train_dir, filename))

for filename in val_mask_filenames:
    shutil.move(os.path.join(masks_dir, filename), os.path.join(masks_val_dir, filename))

for filename in test_mask_filenames:
    shutil.move(os.path.join(masks_dir, filename), os.path.join(masks_test_dir, filename))

print("Files have been split into training, validation, and test sets based on filename")

# Print size of training, validation, and test sets
print(f"Number of training images: {len(train_image_filenames)}")
print(f"Number of validation images: {len(val_image_filenames)}")
print(f"Number of test images: {len(test_image_filenames)}")
