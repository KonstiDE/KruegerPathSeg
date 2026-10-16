import warnings

import scipy.ndimage as resample
import numpy as np
from scipy.interpolate import interpolate

import config.config as cfg

from objects.dataframe import DataFrame


def slice_n_dice(data, mask, aux, t, part):
    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, 0)

    assert data.shape[1:3] == mask.shape[1:3]

    channels, rows, cols = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[:, r:r + t, c:c + t]
            tile_mask = mask[:, r:r + t, c:c + t]
            tile_aux = aux[:, r:r + t, c:c + t]
            class_y = np.max(tile_mask)
            if tile_data.shape[1] != tile_data.shape[2]:
                padded_data = np.zeros((tile_data.shape[0], t, t))
                padded_mask = np.zeros((1, t, t))
                padded_aux = np.zeros((tile_aux.shape[0], t, t))
                padded_data[:, :tile_data.shape[1], :tile_data.shape[2]] = tile_data
                padded_mask[:, :tile_mask.shape[1], :tile_mask.shape[2]] = tile_mask
                padded_aux[:, :tile_aux.shape[1], :tile_aux.shape[2]] = tile_aux

                df = DataFrame(
                    index_w=int(r / t),
                    index_h=int(c / t),
                    data_stack=padded_data,
                    overlay=padded_mask,
                    src_part=part,
                    classification=class_y,
                    auxiliary=padded_aux
                )

            else:
                df = DataFrame(
                    index_w=int(r / t),
                    index_h=int(c / t),
                    data_stack=tile_data,
                    overlay=tile_mask,
                    src_part=part,
                    classification=class_y,
                    auxiliary=tile_aux
                )

            frames.append(df)

    return frames


def raw_slicing(data, t):
    channels, rows, cols = data.shape
    frames = []

    blob_size = cfg.blob_size()

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[:, r:r + t, c:c + t]
            if tile_data.shape[1:] != (blob_size, blob_size):
                padded_data = np.zeros((channels, t, t))
                padded_data[:, :tile_data.shape[1], :tile_data.shape[2]] = tile_data

                frames.append(padded_data)
            else:
                frames.append(tile_data)

    return frames


def check_integrity(data_frames):
    valid_frames = []

    for df in data_frames:
        if not np.all(df.overlay == 0):
            valid_frames.append(df)

    if len(valid_frames) == 0:
        warnings.warn("Variable valid_frame is empty after check_integrity() was called.")

    return valid_frames


def check_array_integrity(array):
    return len(np.unique(array)) > 1


def raw_slicing_no_channels(data, t):
    rows, cols = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t]
            if tile_data.shape != (t, t):
                padded_data = np.zeros((t, t))
                padded_data[:tile_data.shape[0], :tile_data.shape[1]] = tile_data

                frames.append(padded_data)
            else:
                frames.append(tile_data)

    return frames


def pad_image_rgb(img, color, border_width=.1):
    h, w, c = img.shape

    border = int(float(min(h, w) * border_width))
    result = np.full((h + 2 * border, w + 2 * border, c), color, dtype=img.dtype)

    result[border:-border, border:-border] = img

    return result

