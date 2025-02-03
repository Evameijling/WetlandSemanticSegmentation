from PIL import Image

def visualize_image_tiles(image_path, output_image_path, tile_size=(256, 256), spacing=50):
    """
    Creates a visualization of how an image can be split into tiles, including remainder tiles,
    with spacing between the tiles.

    Args:
        image_path (str): Path to the input image file.
        output_image_path (str): Path to save the output visualization.
        tile_size (tuple): Size of each tile, specified as (width, height).
        spacing (int): Space (in pixels) between tiles.
    """
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Tile dimensions
    tile_width, tile_height = tile_size
    tiles_x = (img_width + tile_width - 1) // tile_width  # Number of columns including remainders
    tiles_y = (img_height + tile_height - 1) // tile_height  # Number of rows including remainders

    # Calculate the size of the output visualization
    vis_width = tiles_x * tile_width + (tiles_x - 1) * spacing
    vis_height = tiles_y * tile_height + (tiles_y - 1) * spacing

    # Create a blank (transparent) canvas for the visualization
    visualization = Image.new("RGBA", (vis_width, vis_height), (255, 255, 255, 0))

    # Draw the tiles on the visualization
    for row in range(tiles_y):
        for col in range(tiles_x):
            left = col * (tile_width + spacing)
            upper = row * (tile_height + spacing)

            # Calculate the corresponding region in the original image
            src_left = col * tile_width
            src_upper = row * tile_height
            src_right = min(src_left + tile_width, img_width)
            src_lower = min(src_upper + tile_height, img_height)

            # Crop the region and paste it onto the visualization
            tile = image.crop((src_left, src_upper, src_right, src_lower))
            visualization.paste(tile, (left, upper))

    # Save the visualization
    visualization.save(output_image_path)
    print(f"Visualization saved to {output_image_path}.")

# Example usage:
# make loop of i from range 1 to 26 index
for i in range(1, 27):
    band = f"Band_{i}"

    input_image_path = f"/home/egmelich/SatelliteMAE/Preprocessing_Sentinel2/bands_output/{band}.jpeg"
    output_visualization_path = f"/home/egmelich/SatelliteMAE/Preprocessing_Sentinel2/bands_output/{band}_visualized_tiles.png"
    visualize_image_tiles(input_image_path, output_visualization_path)
