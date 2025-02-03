import ftplib
import configparser
import os
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.transform import from_bounds
import rioxarray
import xarray as x_arr
import re
import zipfile
from tqdm import tqdm
import shutil

# config = configparser.ConfigParser()

# config.read(os.getcwd() + '\\config.cfg', encoding='utf-8')

# ftp_host = config.get('FTP','FTP_HOST')
# ftp_user = config.get('FTP','FTP_USER')
# ftp_pass = config.get('FTP','FTP_PASS')

ftp_host = 'ftp.satellietdataportaal.nl'
ftp_user = 'evameijling'
ftp_pass = 'P()nt4VnjN'
ftp_port = 21

def main():
    # Change to the remote directory
    dir_download = (
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2024/2024_09/20240921/20240921_104931_PNEO-04_2_1_30cm_RD_12bit_RGBNED_Biesbosch.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_09/20230930/20230930_103458_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Made.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_06/20230601/20230601_105654_PNEO-04_1_3_30cm_RD_12bit_RGBNED_Biesbosch.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_04/20230430/20230430_104233_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Sliedrecht.zip'
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_04/20230415/20230415_105231_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Biesbosch.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_12/20231207/20231207_104133_PNEO-03_1_54_30cm_RD_12bit_RGBNED_Lauwersmeer.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2023/2023_09/20230908/20230908_111601_SVNEO-02_30cm_RD_11bit_BGRN_Lauwersmeer.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2023/2023_09/20230907/20230907_105113_SVNEO-01_30cm_RD_11bit_BGRN_Lauwersoog.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2024/2024_08/20240827/20240827_110750_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Hilversum.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2024/2024_01/20240122/20240122_110558_SVNEO-02_30cm_RD_11bit_BGRN_Maarssen.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2024/2024_01/20240122/20240122_110558_SVNEO-02_30cm_RD_11bit_BGRN_Maarssen.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2023/2023_08/20230823/20230823_110234_SVNEO-02_30cm_RD_11bit_BGRN_LoosdrechtsePlassen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_06/20230604/20230604_110438_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Breukelen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_05/20230503/20230503_104955_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Harmelen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2024/2024_05/20240502/20240502_110752_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Oostvaardersplassen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_11/20231114/20231114_104956_PNEO-04_1_16_30cm_RD_12bit_RGBNED_Oostvaardersplassen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_06/20230603/20230603_104552_PNEO-03_1_1_30cm_RD_12bit_RGBNED_Oostvaardersplassen.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_04/20230430/20230430_104201_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Almere.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2023/2023_05/20230527/20230527_104800_SVNEO-01_30cm_RD_11bit_BGRN_MillingenAanDeRijn.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2023/2023_06/20230608/20230608_105312_SVNEO-01_30cm_RD_11bit_BGRN_MillingenAanDeRijn.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2023/2023_06/20230616/20230616_104552_PNEO-04_1_31_30cm_RD_12bit_RGBNED_Yerseke.zip',
        r'/SuperView-NEO/30cm_BGRN_11bit_SVNEO/2024/2024_01/20240111/20240111_111635_SVNEO-01_30cm_RD_11bit_BGRN_Kruispolderhaven.zip',
        r'/Pleiades-NEO/30cm_RGBNED_12bit_PNEO/2024/2024_10/20241022/20241022_104559_PNEO-04_1_1_30cm_RD_12bit_RGBNED_Kruispolderhaven.zip'
    )

    # dir_download = (
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2024/2024_09/20240921/20240921_104931_PNEO-04_2_1_30cm_RD_8bit_RGB_Biesbosch.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_09/20230930/20230930_103458_PNEO-03_1_1_30cm_RD_8bit_RGB_Made.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_06/20230601/20230601_105654_PNEO-04_1_3_30cm_RD_8bit_RGB_Biesbosch.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_04/20230430/20230430_104233_PNEO-04_1_1_30cm_RD_8bit_RGB_Sliedrecht.zip' 
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_04/20230415/20230415_105231_PNEO-04_1_1_30cm_RD_8bit_RGB_Biesbosch.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_12/20231207/20231207_104133_PNEO-03_1_54_30cm_RD_8bit_RGB_Lauwersmeer.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2023/2023_09/20230908/20230908_111601_SVNEO-02_30cm_RD_8bit_RGB_Lauwersmeer.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2023/2023_09/20230907/20230907_105113_SVNEO-01_30cm_RD_8bit_RGB_Lauwersoog.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2024/2024_08/20240827/20240827_110750_PNEO-04_1_1_30cm_RD_8bit_RGB_Hilversum.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2024/2024_01/20240122/20240122_110558_SVNEO-02_30cm_RD_8bit_RGB_Maarssen.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2023/2023_08/20230823/20230823_110234_SVNEO-02_30cm_RD_8bit_RGB_LoosdrechtsePlassen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_06/20230604/20230604_110438_PNEO-03_1_1_30cm_RD_8bit_RGB_Breukelen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_05/20230503/20230503_104955_PNEO-03_1_1_30cm_RD_8bit_RGB_Harmelen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2024/2024_05/20240502/20240502_110752_PNEO-03_1_1_30cm_RD_8bit_RGB_Oostvaardersplassen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_11/20231114/20231114_104956_PNEO-04_1_16_30cm_RD_8bit_RGB_Oostvaardersplassen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_06/20230603/20230603_104552_PNEO-03_1_1_30cm_RD_8bit_RGB_Oostvaardersplassen.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_04/20230430/20230430_104201_PNEO-04_1_1_30cm_RD_8bit_RGB_Almere.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2023/2023_05/20230527/20230527_104800_SVNEO-01_30cm_RD_8bit_RGB_MillingenAanDeRijn.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2023/2023_06/20230608/20230608_105312_SVNEO-01_30cm_RD_8bit_RGB_MillingenAanDeRijn.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2023/2023_06/20230616/20230616_104552_PNEO-04_1_31_30cm_RD_8bit_RGB_Yerseke.zip',
    #     # r'/SuperView-NEO/30cm_RGB_8bit_SVNEO/2024/2024_01/20240111/20240111_111635_SVNEO-01_30cm_RD_8bit_RGB_Kruispolderhaven.zip',
    #     # r'/Pleiades-NEO/30cm_RGB_8bit_PNEO/2024/2024_10/20241022/20241022_104559_PNEO-04_1_1_30cm_RD_8bit_RGB_Kruispolderhaven.zip'
    # )

    # 28992 (EPSG code for the Dutch coordinate system)
    bbox_Biesbosch = (4.701297915124542, 51.807788210303215, 4.894678271732374, 51.70221617252017)
    bbox_Lauwersmeer = (6.1347767252728715, 53.41505344152321, 6.2618237604467595, 53.32434985094335)
    bbbox_Loosdrechtse_Plassen = (5.012687802672872, 52.2138379280471, 5.118876998541708, 52.16069283049822)
    bbox_Oostvaardersplassen = (5.342376433425102, 52.490380779538185, 5.43703566514705, 52.44392886268065)
    bbox_Gelderse_Poort = (5.822095350341954, 51.89465497634194, 6.036601250106514, 51.8448139296371)
    bbox_Verdronken_Land_van_Saeftinghe = (4.102317511229444, 51.3820879601744, 4.21287441013834, 51.331070713961765)

    # Iterate over each file to download and process
    for file in dir_download:
        # Download and unzip the file, obtaining the path to the .tif file
        tif_file_path = download_by_ftp(file)
        
        # Determine the bounding box based on file name
        base_name = os.path.basename(file)
        if re.search(r'(Biesbosch|Made|Sliedrecht)', base_name, re.IGNORECASE):
            clip_raster(bbox_Biesbosch, tif_file_path)
        elif re.search(r'(Lauwersmeer|Lauwersoog)', base_name, re.IGNORECASE):
            clip_raster(bbox_Lauwersmeer, tif_file_path)
        elif re.search(r'(Hilversum|Maarssen|LoosdrechtsePlassen|Breukelen|Harmelen)', base_name, re.IGNORECASE):
            clip_raster(bbbox_Loosdrechtse_Plassen, tif_file_path)
        elif re.search(r'(Oostvaardersplasse|Almere)', base_name, re.IGNORECASE):
            clip_raster(bbox_Oostvaardersplassen, tif_file_path)
        elif re.search(r'(MillingenAanDeRijn)', base_name, re.IGNORECASE):
            clip_raster(bbox_Gelderse_Poort, tif_file_path)
        elif re.search(r'(Yerseke|Kruispolderhaven)', base_name, re.IGNORECASE):
            clip_raster(bbox_Verdronken_Land_van_Saeftinghe, tif_file_path)


def unzip_folder(zip_path):
    """
    Unzips a specified zip file to a target directory named after the zip file,
    and returns the file path of the .tif file within the unzipped folder.

    Parameters:
    zip_path (str): Path to the zip file.

    Returns:
    str: File path of the .tif file within the extracted directory.
    """
    # Define the extraction target directory based on the zip filename
    extract_to = os.path.join(os.path.dirname(zip_path), os.path.splitext(os.path.basename(zip_path))[0])

    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Extract the contents of the zip file to `extract_to`
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Extracting {zip_path} to {extract_to}")
        
        # Loop through each file in the zip archive
        for zip_info in zip_ref.infolist():
            extracted_path = os.path.join(extract_to, os.path.basename(zip_info.filename))
            
            # Extract each file directly into `extract_to`
            if zip_info.is_dir():
                continue  # Skip directories; we're only interested in files

            # Ensure the directory for each file exists
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
            with zip_ref.open(zip_info) as source, open(extracted_path, "wb") as target:
                shutil.copyfileobj(source, target)

    # Locate the .tif file within the extracted folder
    tif_file_path = None
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_file_path = os.path.join(root, file)
                break
        if tif_file_path:
            break

    if not tif_file_path:
        raise ValueError("No .tif file found in the zip archive after extraction.")
    
    print(f"Extraction complete. .tif file located at {tif_file_path}")
    return tif_file_path

def move_contents_up_one_level(directory):
    """
    Moves the contents of a directory up one level if the directory contains a single subdirectory with the same name.

    Parameters:
    directory (str): Path to the directory.
    """
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if len(subdirs) == 1 and subdirs[0] == os.path.basename(directory):
        subdir_path = os.path.join(directory, subdirs[0])
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            shutil.move(item_path, directory)
        os.rmdir(subdir_path)

def download_by_ftp(remote_file_path):
    local_dir = r'C:\Users\eva.gmelich.meijling\OneDrive - Accenture\Documents\AA satelietdataportaal\FTP_downloaded_6bands'
    local_filename = os.path.basename(remote_file_path)
    local_path = os.path.join(local_dir, local_filename)

    # If unzipped folder exists, check if .tif file exists
    extract_to = os.path.join(local_dir, os.path.splitext(local_filename)[0])
    print(f"Checking if unzipped folder exists: {extract_to}")
    if os.path.exists(extract_to):
        # Recursively search for .tif file within extract_to
        for root, _, files in os.walk(extract_to):
            tif_file = next((f for f in files if f.lower().endswith('.tif')), None)
            if tif_file:
                tif_path = os.path.join(root, tif_file)
                print(f"Unzipped folder already exists: {extract_to}. .tif file found at {tif_path}.")
                return tif_path

    # If only the .zip file exists, unzip it to get the .tif file path
    if os.path.exists(local_path):
        print(f"Zip file already exists: {local_path}. Unzipping to get .tif.")
        return unzip_folder(local_path)

    with ftplib.FTP_TLS() as ftp:
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_user, ftp_pass)
        ftp.prot_p()  # Switch to secure data connection
        ftp.cwd(os.path.dirname(remote_file_path))

        # Get the size of the remote file
        remote_file_size = ftp.size(remote_file_path)

        # Initialize the progress bar
        progress_bar = tqdm(total=remote_file_size, unit='B', unit_scale=True, desc=local_filename)

        def progress_callback(data):
            progress_bar.update(len(data))
            local_file_zip.write(data)

        with open(local_path, 'wb') as local_file_zip:
            print(f"Downloading: {remote_file_path}")
            ftp.retrbinary(f'RETR {os.path.basename(remote_file_path)}', progress_callback)

        progress_bar.close()

    # Unzip downloaded file
    local_file = unzip_folder(local_path)
    print(f"Downloaded and unzipped file: {local_file}")

    return local_file

def clip_raster(bbox, raster_file):
    print(f"Clipping raster file: {raster_file} with bbox: {bbox}")
    
    try:
        # Open the raster file
        data_array = rioxarray.open_rasterio(raster_file, chunks=True, lock=False)
        print("Raster file opened successfully.")
        
        # Get the CRS of the raster
        raster_crs = data_array.rio.crs
        print(f"Raster CRS: {raster_crs}")
        
        # Transform the bounding box to the raster CRS
        bbox_transform = rasterio.warp.transform_bounds(
            rasterio.crs.CRS.from_epsg(4326),
            raster_crs,
            bbox[0],
            bbox[3],
            bbox[2],
            bbox[1]
        )
        print(f"Transformed bbox: {bbox_transform}")
        
        # Clip the raster using the transformed bounding box
        raster_clip = data_array.rio.clip_box(
            minx=bbox_transform[0],
            miny=bbox_transform[1],
            maxx=bbox_transform[2],
            maxy=bbox_transform[3],
            crs=raster_crs)
        print("Raster clipped successfully.")
        
        # Define the output directory for clipped files
        clipped_dir = r'C:\Users\eva.gmelich.meijling\OneDrive - Accenture\Documents\AA satelietdataportaal\FTP_downloaded_6bands\clipped_tifs'
        os.makedirs(clipped_dir, exist_ok=True)

        # Define the path for the clipped file
        clipped_file_path = os.path.join(clipped_dir, os.path.basename(raster_file).replace('.tif', '_clipped.tif'))
        
        # Save the clipped raster
        raster_clip.rio.to_raster(clipped_file_path)
        print(f"Clipped raster saved to {clipped_file_path}")

    except Exception as e:
        print(f"Error during clipping: {e}")

if '__main__':
    main()