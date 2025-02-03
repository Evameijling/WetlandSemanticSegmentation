import os
import shutil
import random

# Set a seed for reproducibility (optional)
random.seed(42)

# Input directories
images_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/images_cuts_1024'
masks_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_cuts_1024'

# Output directories
images_train_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_images_train'
images_val_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_images_val'
images_test_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_images_test'

masks_train_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_masks_train'
masks_val_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_masks_val'
masks_test_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP_1024/Biesbosch_masks_test'

# Ensure output directories exist
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(images_test_dir, exist_ok=True)

os.makedirs(masks_train_dir, exist_ok=True)
os.makedirs(masks_val_dir, exist_ok=True)
os.makedirs(masks_test_dir, exist_ok=True)

# Get list of image and mask filenames
image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
mask_filenames = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

# Sort them to ensure consistent ordering
image_filenames.sort()
mask_filenames.sort()

print(f"Found {len(image_filenames)} images and {len(mask_filenames)} masks.")

# Ensure the number of images matches the number of masks
assert len(image_filenames) == len(mask_filenames), "Mismatch between number of images and masks"

# Pair images with corresponding masks
pairs = list(zip(image_filenames, mask_filenames))

# Shuffle the pairs to randomize the split
random.shuffle(pairs)

# Calculate split indices
total_count = len(pairs)
train_count = int(0.75 * total_count)  # 75%
val_count = int(0.15 * total_count)    # 15%
# The rest go to test
test_count = total_count - train_count - val_count

# Split the pairs
train_pairs = pairs[:train_count]
val_pairs = pairs[train_count:train_count + val_count]
test_pairs = pairs[train_count + val_count:]

print(f"Total files: {total_count}")
print(f"Training: {len(train_pairs)}")
print(f"Validation: {len(val_pairs)}")
print(f"Test: {len(test_pairs)}")

# Function to move files to the specified directory
def move_files(pairs_list, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir):
    for img_filename, mask_filename in pairs_list:
        shutil.move(os.path.join(src_images_dir, img_filename),
                    os.path.join(dst_images_dir, img_filename))
        shutil.move(os.path.join(src_masks_dir, mask_filename),
                    os.path.join(dst_masks_dir, mask_filename))

# Move the files
move_files(train_pairs, images_dir, masks_dir, images_train_dir, masks_train_dir)
move_files(val_pairs, images_dir, masks_dir, images_val_dir, masks_val_dir)
move_files(test_pairs, images_dir, masks_dir, images_test_dir, masks_test_dir)

print("Files have been split into training, validation, and test sets based on percentage.")
