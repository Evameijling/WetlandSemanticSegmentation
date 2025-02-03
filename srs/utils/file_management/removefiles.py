import os

# Directories
satellite_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPsatimagery_selectedbands_cuts'
mask_dir = '/projects/0/prjs1235/Satellietdataportaal_data/original_SDPmergedmasks_cuts'

# Search term
search_term = '20240921_104931'

def remove_files(directory, term):
    removed_files_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if term in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                removed_files_count += 1
                print(f"Removed file: {file_path}")
    return removed_files_count

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

# Remove files with the search term in both directories
removed_satellite_files = remove_files(satellite_dir, search_term)
removed_mask_files = remove_files(mask_dir, search_term)

# Count remaining files in both directories
remaining_satellite_files = count_files(satellite_dir)
remaining_mask_files = count_files(mask_dir)

print(f"Removed {removed_satellite_files} files from {satellite_dir}")
print(f"Removed {removed_mask_files} files from {mask_dir}")
print(f"Remaining files in {satellite_dir}: {remaining_satellite_files}")
print(f"Remaining files in {mask_dir}: {remaining_mask_files}")