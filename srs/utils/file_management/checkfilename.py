import os
import rasterio
import numpy as np

# Input directories for satellite images
satellite_dir ='/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery'

# Get sorted list of image filenames
image_filenames = sorted([f for f in os.listdir(satellite_dir) if f.endswith('.tif')])

# Print the filenames
print(f"Number of images: {len(image_filenames)}")
# print(f"Image filenames: {image_filenames}")

# Function to check if the filename contains specific keywords
def is_test_filename(filename):
    return any(keyword in filename.lower() for keyword in ["sliedrecht"])

for img_filename in image_filenames:
    if is_test_filename(img_filename):
    # Print min and max pixel value of the image
        with rasterio.open(os.path.join(satellite_dir, img_filename)) as src:
            img = src.read()
            print(f"Image: {img_filename}")
            print(f"Min pixel value: {np.min(img)}")
            print(f"Max pixel value: {np.max(img)}")
            # Print size of the image
            print(f"Image size: {img.shape}")

##########################################################################################

# input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmasks'

# # Print the min and max pixel value of the file that has 'sliedrecht' in the file name

# # Get sorted list of image filenames
# image_filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])

# # Function to check if the filename contains specific keywords
# def is_test_filename(filename):
#     return any(keyword in filename.lower() for keyword in ["sliedrecht"])

# for img_filename in image_filenames:
#     if is_test_filename(img_filename):
#         # Print min and max pixel value of the image
#         with rasterio.open(os.path.join(input_dir, img_filename)) as src:
#             img = src.read()
#             print(f"Image: {img_filename}")
#             print(f"Min pixel value: {np.min(img)}")
#             print(f"Max pixel value: {np.max(img)}")
#             # Print size of the image
#             print(f"Image size: {img.shape}")

# ##########################################################################################

# # from '/projects/0/prjs1235/Satellietdataportaal_data/images_test' and '/projects/0/prjs1235/Satellietdataportaal_data/masks_test', remove all files that contain 'sliedrecht' in the file name

# images_test_dir = '/projects/0/prjs1235/Satellietdataportaal_data/images_test'
# masks_test_dir = '/projects/0/prjs1235/Satellietdataportaal_data/masks_test'

# # Get sorted list of image and mask filenames
# image_filenames = sorted([f for f in os.listdir(images_test_dir) if f.endswith('.tif')])
# mask_filenames = sorted([f for f in os.listdir(masks_test_dir) if f.endswith('.tif')])
# print(f"Number of images before removal: {len(image_filenames)}")
# print(f"Number of masks before removal: {len(mask_filenames)}")

# # Function to check if the filename contains specific keywords
# def is_test_filename(filename):
#     return any(keyword in filename.lower() for keyword in ["sliedrecht"])

# # Remove files that contain 'sliedrecht' in the file name
# for img_filename, mask_filename in zip(image_filenames, mask_filenames):
#     if is_test_filename(img_filename):
#         os.remove(os.path.join(images_test_dir, img_filename))
#         os.remove(os.path.join(masks_test_dir, mask_filename))

# # Check the number of images and masks after removing the files
# print(f"Number of images after removal: {len(os.listdir(images_test_dir))}")
# print(f"Number of masks after removal: {len(os.listdir(masks_test_dir))}")

# ##########################################################################################

# # print the number of different pixel values in the images of '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmasks' 

# # input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmasks'
# # input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/masks'
# input_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery'


# # Get sorted list of image filenames
# image_filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])

# # print the number of different pixel values in the images
# for img_filename in image_filenames[0:50]:
#     with rasterio.open(os.path.join(input_dir, img_filename)) as src:
#         img = src.read()
#         print(f"Image: {img_filename}")
#         print(f"Number of different pixel values: {len(np.unique(img))}")

    