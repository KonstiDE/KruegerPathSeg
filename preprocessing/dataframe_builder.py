"""
    Script to build up dataframes from big tifs
"""

import os
import warnings

import numpy as np
import config.config as cfg
import rasterio as rio
import rasterio.features as rft
import geopandas as gpd

from preprocessing.tools import slice_n_dice, check_integrity
from preprocessing.export import export_to_npz


def walk(parts, site):
    for part in parts:
        print("==================================================")
        print("Beginning on part {} from site {}".format(part, site))
        vec_paths = {}
        aux_paths = []

        print("Locating features...")

        for feature_name, feature_id in cfg.get_features().items():
            path = os.path.join(cfg.get_raw_data_path(), "{}".format(site), "{}".format(part), feature_name)

            if os.path.isfile(path):
                vec_paths[path] = feature_id
            else:
                warnings.warn("Did not find feature {} for site {}. Ignoring...".format(feature_name, site))

        for raster_name in cfg.get_auxiliary_raster():
            path = os.path.join(cfg.get_raw_data_path(), "{}".format(site), "{}".format(part), raster_name)

            if os.path.isfile(path):
                aux_paths.append(path)
            else:
                warnings.warn("Did not find raster {} for site {}. Ignoring...".format(path, site))

        aux_paths = sorted(aux_paths)


        print("Building overlay...")
        data_stack, gt_overlay, aux_overlay = overlay_python(
            tif_path=os.path.join(cfg.get_raw_data_path(), "{}".format(site), "{}".format(part), "rgb_clipped.tif"),
            vec_paths=vec_paths,
            rast_paths=aux_paths
        )
        print("Slicing...")
        frames = slice_n_dice(data_stack, gt_overlay, aux_overlay, cfg.blob_size(), part)

        print("Checking integrity...")
        frames = check_integrity(frames)

        print("Exporting...")
        export_to_npz(frames=frames, name=site)
        print("==================================================")


def overlay_python(tif_path, vec_paths, rast_paths):
    tif = rio.open(tif_path)
    tif_array = tif.read().astype(np.uint8)

    dim = tif.shape

    track_rasters = []
    aux_rasters = []

    for (track_file_path, class_id) in vec_paths.items():
        track_file_entry = gpd.read_file(track_file_path)
        geom = [shapes for shapes in track_file_entry.geometry]

        where = rft.rasterize(
            geom,
            out_shape=dim,
            transform=tif.transform,
            fill=0,
            all_touched=True,
            default_value=1
        )

        track_rasters.append(where)

    for c in range(len(rast_paths)):
        raster_file_path = rast_paths[c]
        raster_file_entry = rio.open(raster_file_path).read().astype(np.float16)

        aux_rasters.append(raster_file_entry)

    overlay_raster = np.sum(np.stack(track_rasters), axis=0).astype(np.uint8)
    overlay_raster[overlay_raster > 1] = 1

    if len(aux_rasters) > 0:
        aux_rasters = np.stack(aux_rasters)

        if len(aux_rasters.shape) > 3:
            aux_rasters = aux_rasters.squeeze(1)

    if len(tif.indexes) > 3:
        tif_array = tif_array[0:3, :, :]

    # if len(input_rasters) > 0:
    #     if len(input_rasters) == 1:
    #         input_additionals = np.dstack(input_rasters)
    #     else:
    #         input_additionals = np.stack(input_rasters, axis=0).squeeze(1)
    #
    #     tif_array = np.concatenate((tif_array, input_additionals))

    # numpy transpose
    # plt.imshow(rgb.transpose((1, 2, 0)))
    # plt.show()

    return tif_array, overlay_raster, aux_rasters

