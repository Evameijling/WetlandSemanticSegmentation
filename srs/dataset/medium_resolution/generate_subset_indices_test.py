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
        "train": 1701,
        "val": 948,
        "test": 2280  # Original size of the test set
    }

    # Fractions for subsets
    fractions = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]  # Fractions of the dataset (e.g., 10%, 50%, 100%)
    seed = 42  # Fixed seed for reproducibility

    # Redefine the test set size
    # reduced_test_size = 948
    reduced_test_size = 798
    np.random.seed(seed)
    test_indices = np.random.choice(splits["test"], reduced_test_size, replace=False)

    # Generate indices for each split and fraction
    subsets = {}
    for split, dataset_size in splits.items():
        if split == "test":
            dataset_size = reduced_test_size  # Use reduced test size
            split_indices = test_indices  # Use preselected test indices
        else:
            split_indices = list(range(dataset_size))
        
        subsets[split] = {
            f"{int(fraction * 100)}%": create_subset_indices(len(split_indices), fraction, seed)
            for fraction in fractions
        }

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the output file path in the same directory
    output_file = os.path.join(script_dir, "subset_indices_per_split_test.json")

    # Save to a JSON file
    with open(output_file, "w") as f:
        json.dump(subsets, f, indent=4)

    print(f"Subset indices saved to {output_file}")

if __name__ == "__main__":
    main()
