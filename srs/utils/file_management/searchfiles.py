import os

# Directory to search
search_dir = '/projects/0/prjs1235/Satellietdataportaal_data/images_test'

# Search term
# search_term = '20230430_104233'
search_term = '20240921_104931'

def search_files(directory, term):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if term in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

# Run the search
matching_files = search_files(search_dir, search_term)

# Print the results
if matching_files:
    print(f"Found {len(matching_files)} file(s) with '{search_term}' in their name:")
    # for file in matching_files:
        # print(file)
else:
    print(f"No files found with '{search_term}' in their name.")