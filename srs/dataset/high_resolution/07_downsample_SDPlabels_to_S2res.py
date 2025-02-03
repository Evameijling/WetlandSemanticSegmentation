#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
from PIL import Image

import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling

def downsample_png_mask(
    mask_png_path: str,
    georef_tif_path: str,
    sentinel2_tif_path: str,
    output_png_path: str,
    resampling_method: Resampling = Resampling.nearest
):
    """
    Downsample a PNG mask (no georeference) by:
      1) Loading its corresponding GeoTIFF's metadata (same name, same dims).
      2) Creating an in-memory georeferenced raster from the PNG.
      3) Reprojecting/resampling to match Sentinel-2 geometry (10 m).
      4) Saving the result to a new PNG (same pixel size as Sentinel-2).

    Note: The output PNG won't have embedded georeferencing. We're only
    preserving pixel alignment for segmentation tasks.
    """

    # ------------------------------------------------------------------
    # 1) Read the PNG as a NumPy array
    # ------------------------------------------------------------------
    # If your mask is single-band, we can convert to "L" for 8-bit grayscale.
    # If it's already grayscale, you can skip the convert('L').
    mask_img = Image.open(mask_png_path).convert('L')
    mask_data = np.array(mask_img)
    # mask_data.shape => (height, width)

    # ------------------------------------------------------------------
    # 2) Read the georeference from the corresponding GeoTIFF
    # ------------------------------------------------------------------
    # This .tif has the same pixel dimensions as the PNG, plus transform + CRS.
    with rasterio.open(georef_tif_path) as src_georef:
        # Copy its metadata
        georef_meta = src_georef.meta.copy()
        georef_transform = src_georef.transform
        georef_crs = src_georef.crs
        
    # Verify the dimensions match (optional but recommended)
    if (georef_meta['width'], georef_meta['height']) != (mask_data.shape[1], mask_data.shape[0]):
        raise ValueError(
            f"Mismatch in dimensions between PNG mask ({mask_data.shape}) "
            f"and georeference TIF ({georef_meta['width']}x{georef_meta['height']})."
        )

    # We'll create an in-memory dataset for the PNG data, using the TIF's metadata.
    georef_meta.update({
        'count': 1,                           # single-band mask
        'dtype': mask_data.dtype.name,        # e.g. uint8
        'driver': 'GTiff'                     # in-memory GeoTIFF
    })

    # ------------------------------------------------------------------
    # 3) Create an in-memory raster for the PNG
    # ------------------------------------------------------------------
    with MemoryFile() as memfile:
        with memfile.open(**georef_meta) as mem:
            # Write the mask data as band 1
            mem.write(mask_data, 1)

            # Now `mem` is a georeferenced version of the PNG mask.

            # ------------------------------------------------------------------
            # 4) Reproject to match Sentinel-2 geometry (10 m)
            # ------------------------------------------------------------------
            with rasterio.open(sentinel2_tif_path) as ref_sen:
                ref_transform = ref_sen.transform
                ref_crs = ref_sen.crs
                ref_width = ref_sen.width
                ref_height = ref_sen.height

                # Prepare output array for the reprojected data
                out_data = np.empty((ref_height, ref_width), dtype=mask_data.dtype)

                reproject(
                    source=rasterio.band(mem, 1),
                    destination=out_data,
                    src_transform=georef_transform,
                    src_crs=georef_crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method
                )

    # ------------------------------------------------------------------
    # 5) Save result as PNG
    # ------------------------------------------------------------------
    # out_data is now aligned with the S2 grid. We store as an 8-bit PNG.
    # If your mask values are beyond 0-255, you might need to scale or use a different mode.
    downsampled_img = Image.fromarray(out_data)
    downsampled_img.save(output_png_path)

    print(f"Created {output_png_path}, shape = {out_data.shape}, resampling={resampling_method}")


def main():
    """
    Example usage of downsample_png_mask.
    We assume:
    1) mask_png_path is a single-band label .png in the "masks" folder.
    2) georef_tif_path is the .tif in "annotation_images_selectedbands" with the same name, 
       which contains the needed geotransform/CRS.
    3) sentinel2_tif_path is the 10 m Sentinel-2 that was clipped to the same area.
    4) output_png_path is where we save the downsampled mask.

    Adjust paths as needed!
    """
    file = '20230415_105231_Biesbosch_tile_16'
    sen2_tif = 'Biesbosch_Sentinel2_2023-09-09_clip_20230415_105231_Biesbosch_tile_16.tif'

    # (1) The original mask (no georeference)
    mask_png_path = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks/{file}.png"
    
    # (2) The georeferenced TIF with the same name
    #     (it lives in annotation_images_selectedbands, same tile name)
    georef_tif_path = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/imagery_tiles/annotation_images_selectedbands/{file}.tif"

    # (3) The corresponding clipped Sentinel-2 TIF (already at 10 m, bounding box aligned)
    sentinel2_tif_path = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles/{sen2_tif}"
    
    # (4) The output path for the downsampled PNG
    output_png_path = f"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/masks_S2downsampled/{file}.png"

    # Run the function
    downsample_png_mask(
        mask_png_path=mask_png_path,
        georef_tif_path=georef_tif_path,
        sentinel2_tif_path=sentinel2_tif_path,
        output_png_path=output_png_path,
        resampling_method=Resampling.nearest
    )

if __name__ == "__main__":
    main()
