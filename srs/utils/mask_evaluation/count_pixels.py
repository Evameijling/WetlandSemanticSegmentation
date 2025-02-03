import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# Function to calculate class distribution directly from a folder
def calculate_class_distribution(mask_dir, num_classes, exclude_class=0):
    """
    Calculate the number of pixels per class in a folder of mask images,
    explicitly excluding a specific class (e.g., "Negative").

    Args:
        mask_dir (str): Path to the directory containing mask images.
        num_classes (int): Total number of classes.
        exclude_class (int): Class index to exclude from the counts.

    Returns:
        class_counts (list): Number of pixels for each class.
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for mask_file in tqdm(sorted(os.listdir(mask_dir)), desc=f"Processing {mask_dir}"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))

        # Flatten the mask and exclude invalid pixels and the excluded class
        valid_pixels = mask[(mask >= 0)]  # Exclude invalid pixels (-1)
        valid_pixels = valid_pixels[valid_pixels != exclude_class]  # Exclude the specified class
        total_pixels += valid_pixels.size
        counts = np.bincount(valid_pixels, minlength=num_classes)
        class_counts += counts

    # Set the count for the excluded class to 0 to ensure it is ignored
    class_counts[exclude_class] = 0

    return class_counts, total_pixels

# Main script logic
if __name__ == "__main__":
    NUM_CLASSES = 9  # Define the number of classes

    # Directories containing mask images
    train_masks_dir ='/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP/Biesbosch_masks_train'
    val_masks_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP/Biesbosch_masks_val'
    test_masks_dir = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/Biesbosch_trainvaltest_SDP/Biesbosch_masks_test'

    # Class names
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

    print("Calculating class distributions...")

    # Calculate class distributions for train, val, and test sets
    train_class_counts, train_total_pixels = calculate_class_distribution(train_masks_dir, NUM_CLASSES)
    val_class_counts, val_total_pixels = calculate_class_distribution(val_masks_dir, NUM_CLASSES)
    test_class_counts, test_total_pixels = calculate_class_distribution(test_masks_dir, NUM_CLASSES)

    # Print class distributions
    print("\nClass Distribution in Train Set:")
    for i, count in enumerate(train_class_counts):
        print(f"{class_names[i]}: {count} pixels ({(count / train_total_pixels) * 100:.2f}%)")

    print("\nClass Distribution in Validation Set:")
    for i, count in enumerate(val_class_counts):
        print(f"{class_names[i]}: {count} pixels ({(count / val_total_pixels) * 100:.2f}%)")

    print("\nClass Distribution in Test Set:")
    for i, count in enumerate(test_class_counts):
        print(f"{class_names[i]}: {count} pixels ({(count / test_total_pixels) * 100:.2f}%)")
