import os
import numpy as np
import json

def create_subset_indices(dataset_size, fraction, seed=42):
    """
    Generate random indices for a subset of the dataset.

    Args:
        dataset_size (int): Total size of the dataset.
        fraction (float): Fraction of the dataset to use (e.g., 0.1 for 10%).
        seed (int): Random seed for reproducibility.

    Returns:
        list: Indices for the subset.
    """
    np.random.seed(seed)
    indices = np.random.choice(dataset_size, int(dataset_size * fraction), replace=False)
    return indices.tolist()  # Convert to list

def main():
    # Number of images in each split
    splits = {
        "train": 1027,
        "val": 205,
        "test": 138
    }
    
    # Found 1634 images and 1634 masks.
    # Total files: 1634
    # Training: 1225
    # Validation: 245
    # Test: 164
    # Files have been split into training, validation, and test sets based on percentage.

    # Fractions for subsets
    fractions = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]  # Fractions of the dataset (e.g., 10%, 50%, 100%)
    seed = 42  # Fixed seed for reproducibility

    # Generate indices for each split and fraction
    subsets = {}
    for split, dataset_size in splits.items():
        subsets[split] = {
            f"{int(fraction * 100)}%": create_subset_indices(dataset_size, fraction, seed)
            for fraction in fractions
        }

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the output file path in the same directory
    output_file = os.path.join(script_dir, "subset_indices_per_split_SDP_1024.json")

    # Save to a JSON file
    with open(output_file, "w") as f:
        json.dump(subsets, f, indent=4)

    print(f"Subset indices saved to {output_file}")

if __name__ == "__main__":
    main()
