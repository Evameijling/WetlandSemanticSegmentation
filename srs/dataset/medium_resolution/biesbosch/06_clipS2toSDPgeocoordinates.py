import os
from pathlib import Path

import rasterio
import rasterio.mask
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

def main():
    hr_dir = r"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/imagery_tiles/annotation_images"
    lr_tif_path = r"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery/selectedbands/Biesbosch_Sentinel2_2023-09-09.tif"
    output_dir = r"/projects/0/prjs1235/Satellietdataportaal_data/Biesbosch_ManualAnnotation/S2_imagery_tiles"

    os.makedirs(output_dir, exist_ok=True)

    # Open LR once to read metadata
    with rasterio.open(lr_tif_path) as lr_src:
        lr_crs = lr_src.crs

    hr_dir_path = Path(hr_dir)
    hr_tif_paths = sorted(hr_dir_path.glob("*.tif"))

    # ----- Get the low-res base name for output naming -----
    lr_base_name = Path(lr_tif_path).stem

    for hr_path in hr_tif_paths:
        print(f"\nProcessing HR tile: {hr_path.name}")

        # 1) Get HR tile bounds + CRS
        with rasterio.open(hr_path) as hr_src:
            hr_bounds = hr_src.bounds
            hr_crs = hr_src.crs

        # 2) Reproject bounds if CRS differ
        if hr_crs != lr_crs:
            reproj_bounds = transform_bounds(
                src_crs=hr_crs,
                dst_crs=lr_crs,
                left=hr_bounds.left,
                bottom=hr_bounds.bottom,
                right=hr_bounds.right,
                top=hr_bounds.top
            )
            bbox_in_lr = BoundingBox(*reproj_bounds)
        else:
            bbox_in_lr = hr_bounds

        # 3) Clip LR to bounding box
        geom_box = box(bbox_in_lr.left, bbox_in_lr.bottom, bbox_in_lr.right, bbox_in_lr.top)
        geom_geojson = [mapping(geom_box)]

        # ----- Here is where we change the output name -----
        # e.g. "Biesbosch_Sentinel2_2023-09-09_clip_20230430_tile_1.tif"
        out_name = f"{lr_base_name}_clip_{hr_path.stem}.tif"
        out_path = os.path.join(output_dir, out_name)

        with rasterio.open(lr_tif_path) as lr_src:
            out_image, out_transform = rasterio.mask.mask(
                lr_src,
                shapes=geom_geojson,
                crop=True,
                nodata=lr_src.nodata,
                all_touched=True
            )
            out_meta = lr_src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": lr_crs
            })

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(out_image)

        print(f"  Saved clipped LR tile to: {out_path}")

    print("\nDone! Created LR clips for each HR tile.")

if __name__ == "__main__":
    main()
