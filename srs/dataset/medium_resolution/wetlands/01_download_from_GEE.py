import ee
import time

# Authenticate and initialize Earth Engine
ee.Initialize(project='evameijling')

# Define the areas of interest with their corresponding geometries
areas_of_interest = {
    'Biesbosch': ee.Geometry.Polygon(
        [
            [
                [4.638834757267918,51.69035434099774],
                [4.926539225041355,51.69035434099774],
                [4.926539225041355,51.81999428517475],
                [4.638834757267918,51.81999428517475]
            ]
        ], None, False
    ),
    'Loosdrecht': ee.Geometry.Polygon(
        [
            [
                [5.008009384029388,52.15951296444678],
                [5.144651840084076,52.15951296444678],
                [5.144651840084076,52.31887414541911],
                [5.008009384029388,52.31887414541911],
                [5.008009384029388,52.15951296444678]
            ]
        ], None, False
    ),
    'Lauwersmeer': ee.Geometry.Polygon(
        [
            [
                [6.104208981238335,53.32004597852978],
                [6.293723141394585,53.32004597852978],
                [6.293723141394585,53.419190268072754],
                [6.104208981238335,53.419190268072754]
            ]
        ], None, False
    ),
    'LandVanSaeftinghe': ee.Geometry.Polygon(
        [
            [
                [4.075397577451039,51.32732670069724],
                [4.207576837704945,51.32732670069724],
                [4.207576837704945,51.380499830741726],
                [4.075397577451039,51.380499830741726]
            ]
        ], None, False
    ),
    'GendtsePolder': ee.Geometry.Polygon(
        [
            [
                [5.8280535419588375,51.84777501802643],
                [6.038853712857275,51.84777501802643],
                [6.038853712857275,51.89398659543555],
                [5.8280535419588375,51.89398659543555]
            ]
        ], None, False
    ),
    'Oostvaardersplassen': ee.Geometry.Polygon(
        [a
            [
                [5.336149906109031,52.447984620281254],
                [5.431936954448875,52.447984620281254],
                [5.431936954448875,52.49566752191786],
                [5.336149906109031,52.49566752191786]
            ]
        ], None, False
    )
}

# Define the tile IDs to be ignored for each area of interest
tiles_to_ignore = {
    'Biesbosch': [],
    'Loosdrecht': ['31UFU'],
    'Lauwersmeer': ['31UFV', '31UGV'],
    'LandVanSaeftinghe': ['31UET'],
    'GendtsePolder': ['31UGT'],
    'Oostvaardersplassen': ['31UFV']
}

# Define the date range
start_date = '2020-01-01'
end_date = '2024-11-01'

# Function to export each matching image pair with aligned clipping
def export_images(aio, geometry):
    # Define collections and filters, and select only RGB and NIR bands
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
        .filter(ee.Filter.inList('MGRS_TILE', tiles_to_ignore.get(aio, [])).Not()) \
        # .select(['B2', 'B3', 'B4', 'B8'])  # Select RGB and NIR bands

    dw_collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)

    def export_single_image(s2_image):
        date = ee.Date(s2_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        # Filter Dynamic World collection to find the corresponding image
        dw_image = dw_collection.filter(ee.Filter.eq('system:index', s2_image.get('system:index'))).first()

        if dw_image:
            # Define the projection based on Sentinel-2
            projection = s2_image.select('B2').projection()

            # Convert both images to UInt16 for compatibility and clip with bounding box
            s2_image_uint16 = s2_image.clip(geometry).reproject(projection).toUint16()
            dw_image_uint16 = dw_image.select('label').clip(geometry).reproject(projection).toUint16()

            # Export Sentinel-2 image with selected bands
            task_s2 = ee.batch.Export.image.toDrive(
                image=s2_image_uint16,
                description=f'{aio}_Sentinel2_{date}',
                folder='Sentinel2_DynamicWorld',
                fileNamePrefix=f'{aio}_Sentinel2_{date}',
                region=geometry.coordinates().getInfo(),
                scale=10,
                maxPixels=1e10
            )
            task_s2.start()
            
            # Export corresponding Dynamic World image
            task_dw = ee.batch.Export.image.toDrive(
                image=dw_image_uint16,
                description=f'{aio}_DynamicWorld_{date}',
                folder='Sentinel2_DynamicWorld',
                fileNamePrefix=f'{aio}_DynamicWorld_{date}',
                region=geometry.coordinates().getInfo(),
                scale=10,
                maxPixels=1e10
            )
            task_dw.start()
            print(f'Export tasks started for {aio} on date: {date}')
        else:
            print(f'No Dynamic World image found for {aio} on date: {date}')

    # Iterate over each Sentinel-2 image and export if it has a matching Dynamic World image
    s2_images = s2_collection.getInfo()['features']
    for s2_image_info in s2_images:
        s2_image = ee.Image(s2_image_info['id'])
        export_single_image(s2_image)
        time.sleep(1)  # Small delay to avoid overwhelming Earth Engine with requests


# Loop over each area of interest and export images
for aio, geometry in areas_of_interest.items():
    export_images(aio, geometry)
    time.sleep(1)  # Add a delay to avoid hitting request limits

print("Export tasks started.")

####################################################################################################

# # Download from Google Drive using gdown instead of the Drive API

# import gdown
# import os
# import argparse

# # Set up argument parsing
# parser = argparse.ArgumentParser(description="Download a Google Drive folder using gdown.")
# parser.add_argument("folder_url", type=str, help="The full URL of the Google Drive folder to download")
# parser.add_argument("--download_path", type=str, default="/projects/0/prjs1235/DynamicWorld_GEEData/original",
#                     help="Custom download directory (default: /projects/0/prjs1235/DynamicWorld_GEEData/original)")
# args = parser.parse_args()

# # Use provided folder URL and download path
# folder_url = args.folder_url
# download_path = args.download_path
# os.makedirs(download_path, exist_ok=True)

# # Download folder to the specified path
# gdown.download_folder(folder_url, output=download_path, quiet=False, use_cookies=False)
