import os

def count_files(directory):
    try:
        # Get the list of files in the directory
        files = os.listdir(directory)
        
        # Count the number of files
        num_files = len(files)
        
        print(f"Number of files in '{directory}': {num_files}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specify the directory to count files in
    directory = '/projects/0/prjs1235/Satellietdataportaal_data/masks_cuts'
    
    # Count the number of files in the directory
    count_files(directory)