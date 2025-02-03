import os

def list_file_names(directory, output_file):
    try:
        # Get the list of files in the directory
        files = os.listdir(directory)
        
        # Write the file names to the output file
        with open(output_file, 'w') as f:
            for file in files:
                f.write(file + '\n')
        
        print(f"File names have been written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Directory to list files from
    directory = '/projects/0/prjs1235/DynamicWorld_GEEData/original'
    
    # Output file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'file_names.txt')
    
    # List the file names in the directory and write to the output file
    list_file_names(directory, output_file)
