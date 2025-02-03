import argparse
import rasterio
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize bands from a .tif file.")
    parser.add_argument("tif_file", type=str, help="Path to the .tif file")
    parser.add_argument("--band", type=int, help="Band number to visualize (1-based index). If not provided, all bands will be visualized.")
    parser.add_argument("--output", type=str, help="Path to save the visualization")
    return parser.parse_args()

def print_metadata(tif_file):
    with rasterio.open(tif_file) as src:
        # Get metadata, CRS, transform, width, height, and number of bands from the TIFF file
        meta = src.meta
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        bbox = src.bounds
        pixel_width = transform[0]
        pixel_height = abs(transform[4])  # Use abs to get the positive value
        num_bands = src.count

        # Print the retrieved information
        print("Metadata:", meta)
        print("CRS:", crs)
        print("Transform:", transform)
        print("Width:", width)
        print("Height:", height)
        print("Bounding Box:", bbox)
        print("Resolution (Width):", pixel_width)
        print("Resolution (Height):", pixel_height)
        print("Number of Bands:", num_bands)

def visualize_band(tif_file, band, output=None):
    with rasterio.open(tif_file) as src:
        # Read the specified band
        band_data = src.read(band)
        
        # Plot the band data
        plt.figure(figsize=(10, 10))
        plt.imshow(band_data, cmap='gray')
        plt.title(f'Band {band} of {tif_file}')
        plt.colorbar()
        
        # Save the plot if output path is provided
        if output:
            plt.savefig(output)
        else:
            plt.show()

def visualize_all_bands(tif_file, output=None):
    with rasterio.open(tif_file) as src:
        num_bands = src.count
        fig, axes = plt.subplots(1, num_bands, figsize=(15, 5))
        
        for band in range(1, num_bands + 1):
            band_data = src.read(band)
            ax = axes[band - 1]
            ax.imshow(band_data, cmap='gray')
            ax.set_title(f'Band {band}')
            ax.axis('off')
        
        plt.suptitle(f'All Bands of {tif_file}')
        
        # Save the plot if output path is provided
        if output:
            plt.savefig(output)
        else:
            plt.show()

def main():
    args = parse_arguments()
    print_metadata(args.tif_file)
    if args.band:
        visualize_band(args.tif_file, args.band, args.output)
    else:
        visualize_all_bands(args.tif_file, args.output)

if __name__ == "__main__":
    main()

# Run the script with the following command:
# python visualize_tif.py /path/to/your.tif --band 1 --output /path/to/save/visualization.png  # To visualize a specific band
# python visualize_tif.py /path/to/your.tif --output /path/to/save/visualization.png           # To visualize all bands