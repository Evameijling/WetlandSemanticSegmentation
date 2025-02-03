import os
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Specify your folder ID and download path
folder_id = "1R4aP-oiCtkQ7TU9i8K8qb1uHQw_ZOjtO"  # Replace with your actual Google Drive folder ID
download_path = "/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery"
os.makedirs(download_path, exist_ok=True)

# Authenticate and initialize the Drive API
credentials = service_account.Credentials.from_service_account_file('/home/egmelich/SatelliteMAE/Preprocessing_Sentinel2/evameijling-a916bad4fda2.json')
service = build('drive', 'v3', credentials=credentials)

def download_file(file_id, file_name, destination_folder):
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(destination_folder, file_name)

    with open(file_path, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {file_name}: {int(status.progress() * 100)}% complete.")

def list_files_in_folder(folder_id):
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
    files = []
    page_token = None

    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields="nextPageToken, files(id, name)",
                                        pageToken=page_token).execute()
        files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return files

# Get and download files in the folder
files = list_files_in_folder(folder_id)
if not files:
    print("No files found in the specified folder.")
else:
    print(f"Found {len(files)} files in the folder. Starting download...")
    for file in files:
        download_file(file['id'], file['name'], download_path)

print("Download completed.")
