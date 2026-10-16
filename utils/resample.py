import numpy as np
import scipy.ndimage
import rasterio as rio
from rasterio.enums import Resampling


def resample(path):
    xres, yres = 0.02019999999999941981, 0.02019999999999941981

    with rio.open(path) as dataset:
        scale_factor_x = dataset.res[0] / xres
        scale_factor_y = dataset.res[1] / yres

        profile = dataset.profile.copy()

        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor_y),
                int(dataset.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear
        )

        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )
        profile.update({
            "height": data.shape[-2],
            "width": data.shape[-1],
            "transform": transform,
            "dtype": rio.uint8,
            "nodata": 0,
        })

    with rio.open("output.tif", "w", **profile) as dataset:
        dataset.write(data.astype(rio.uint8))


if __name__ == '__main__':
    resample("E:/master/koger/airstrip/airstrip_ne_merge_transparent_mosaic_group1__Jessie_raster_scale_2.44140625.jpg_mosaic_crs.tif")

