import os
import shutil
from tqdm import tqdm

# Define source and destination folders and the search string
src_folder = '/projects/0/prjs1235/Satellietdataportaal_data/masks_test'
dst_folder = '/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_masks_cuts'
search_string = '20240921_104931'

def copy_files_with_string(src_folder, dst_folder, search_string):
    # Create destination folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # Get list of files in the source folder
    files = os.listdir(src_folder)
    
    # Iterate over all files in the source folder with a progress bar
    for filename in tqdm(files, desc="Copying files"):
        # Check if the search string is in the filename
        if search_string in filename:
            # Construct full file path
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            # Copy file to the destination folder
            shutil.copy(src_file, dst_file)
            # print(f"Copied: {src_file} to {dst_file}")

# Call the function to copy files
copy_files_with_string(src_folder, dst_folder, search_string)