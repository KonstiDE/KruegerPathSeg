import numpy as np
import scipy.ndimage
import rasterio as rio
from rasterio.enums import Resampling


def comprimise(path):
    with rio.open(path) as dataset:

        array = dataset.read()
        array[array < 0] = 0

        profile = dataset.profile.copy()

        profile.update({
            "dtype": rio.uint8,
            "count": 3
        })

    with rio.open("E:/master/svalbard/rgb_clipped.tif", "w+", **profile) as dataset:
        dataset.write(array.astype(rio.uint8)[0:3, :, :])


if __name__ == '__main__':
    comprimise("E:/master/svalbard/rgb_clipped_7channels.tif")

