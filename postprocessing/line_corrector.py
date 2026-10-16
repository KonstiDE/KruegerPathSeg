from math import sqrt, acos, pi

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import rasterio as rio
import skimage

from PIL import Image
from scipy.signal import convolve2d
from scipy.spatial import KDTree

from scipy.ndimage import binary_hit_or_miss

import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm
from skimage.graph import route_through_array

from postprocessing.models.connection import Connection
from preprocessing.tools import raw_slicing_no_channels


def merge_points(coordinates, d_max):
    tree = KDTree(coordinates)

    visited = np.zeros(len(coordinates), dtype=bool)
    merged_points = []

    for i, point in enumerate(coordinates):
        if not visited[i]:
            indices = tree.query_ball_point(point, d_max)

            if len(indices) > 1:
                cluster_points = []

                for idx in indices:
                    cluster_points.append(coordinates[idx])

                merged_point = np.mean(cluster_points, axis=0).astype(int)

                merged_points.append(tuple(merged_point))

                for idx in indices:
                    visited[idx] = True

            else:
                merged_points.append(coordinates[i])
                visited[i] = True

    return merged_points


def get_counterparts(bubbles, merged_points, d_max_counter, threshold=0.5):
    lookup_table = {}
    visited = []

    bubbles = (bubbles > threshold).astype(np.uint8)

    inverted_cost_array = 1 - bubbles

    # plt.imshow(inverted_cost_array)
    # plt.show()

    for start in merged_points:
        start = tuple(start)[::-1]

        for end in merged_points:
            end = tuple(end)[::-1]

            if np.all(start == end):
                continue

            if np.linalg.norm(np.array(start) - np.array(end)) > d_max_counter:
                continue

            if str(start) in lookup_table.keys() or str(end) in lookup_table.keys():
                continue

            if str(start) + str(end) in visited:
                continue

            try:
                connection, cost = route_through_array(
                    inverted_cost_array,
                    start=start,
                    end=end,
                    fully_connected=True,
                    geometric=False
                )

                visited.append(str(start) + str(end))
                visited.append(str(end) + str(start))

                if cost > 1:
                    continue

                lookup_table[str(start)] = end
                lookup_table[str(end)] = start

            except ValueError:
                pass

    return lookup_table


def calculate_vector(p1, p2):
    return np.array(p2) - np.array(p1)


def calculate_angle(vector1, vector2):
    v1_norm = vector1 / np.linalg.norm(vector1)
    v2_norm = vector2 / np.linalg.norm(vector2)

    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    angle = np.degrees(np.arccos(dot_product))

    return min(angle, 180 - angle)


def angle_between_lines(line1_points, line2_points):
    vector1 = calculate_vector(line1_points[0], line1_points[1])
    vector2 = calculate_vector(line2_points[0], line2_points[1])

    return calculate_angle(vector1, vector2)


def correct(probability, skeleton, d_max, d_max_merge, d_max_counter, max_cost):
    kernels = [
        np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 1, 0]]
        ),
        np.array(
            [[0, 0, 1],
             [0, 1, 0],
             [0, 0, 0]]
        ),
    ]

    hits = []

    for i in range(4):
        hits.extend(np.argwhere(binary_hit_or_miss(skeleton, np.rot90(kernels[0], k=i))))
        hits.extend(np.argwhere(binary_hit_or_miss(skeleton, np.rot90(kernels[1], k=i))))

    for i in range(len(hits)):
        hits[i] = hits[i][::-1]

    if len(hits) > 0:
        merged_points = merge_points(hits, d_max=d_max_merge)

        bubbles = probability.copy()
        for point in merged_points:
            bubbles = cv2.circle(bubbles, point, 30, (1), -1)

        counter_parts = get_counterparts(bubbles, merged_points, d_max_counter=d_max_counter)

        cost_array = -np.log((probability + 1e-6) / (1 + 1e-6))

        connected = []

        for start in merged_points:
            start = tuple(start)[::-1]

            connection_from_here = []

            for end in merged_points:
                end = tuple(end)[::-1]

                true_angle_diff = None

                if np.all(start == end):
                    continue

                if np.linalg.norm(np.array(start) - np.array(end)) > d_max:
                    continue

                if str(start) + str(end) in connected or str(end) + str(start) in connected:
                    continue

                if str(start) in counter_parts.keys():
                    if counter_parts[str(start)] == end:
                        continue

                    true_angle_diff = angle_between_lines((start, counter_parts[str(start)]), (start, end))

                    if true_angle_diff > 60:
                        continue

                # Never goes in here, I think
                if str(end) in counter_parts.keys():
                    if counter_parts[str(end)] == start:
                        continue

                try:
                    connection, cost = route_through_array(
                        cost_array,
                        start=start,
                        end=end,
                        fully_connected=True,
                        geometric=True
                    )

                    if cost > max_cost:
                        continue

                    true_angle_diff = true_angle_diff if true_angle_diff is not None else np.Inf

                    connection_from_here.append(Connection(
                        start,
                        end,
                        connection,
                        cost,
                        true_angle_diff,
                        10 / true_angle_diff + max_cost / cost,
                    ))

                except ValueError:
                    pass



            for connection in sorted(connection_from_here, key=lambda x: x.true_angle_diff, reverse=True)[:3]:
                for row, col in connection.path:
                    skeleton[row, col] = 1

                connected.append(str(connection.start) + str(connection.end))
                connected.append(str(connection.end) + str(connection.start))

        return skeleton

    else:
        return np.zeros_like(skeleton)


def connect_animal_paths():
    variants = ["denseunet"]

    PIL.Image.MAX_IMAGE_PIXELS = None

    threshold = 0.7
    max_distance = 500
    max_distance_merge = 50
    max_distance_counter = 500
    max_cost = 1300

    for variant in variants:
        img_path = "E:/master/part8/mosaic_attention_buffer.tif"

        tif = rio.open(img_path)
        array = np.asarray(Image.open(img_path))
        plt.imshow(array)
        plt.show()

        offset_array = np.pad(array, [256, 0], mode='constant')

        window_size = 4096

        mosaic_width_index = (offset_array.shape[0] // window_size) + 1
        mosaic_height_index = (offset_array.shape[1] // window_size) + 1

        arrays = raw_slicing_no_channels(offset_array, t=window_size)

        result_image = Image.new("F", (mosaic_height_index * window_size, mosaic_width_index * window_size))

        cw = 0
        ch = 0
        for data in tqdm(arrays):
            thresholded = (data >= threshold).astype(np.uint8)

            area_removed = skimage.morphology.area_opening(thresholded, area_threshold=500, connectivity=2)

            skeleton = skeletonize(area_removed).astype(np.uint8)

            paths_corrected = correct(
                data,
                skeleton,
                d_max=max_distance,
                d_max_merge=max_distance_merge,
                d_max_counter=max_distance_counter,
                max_cost=max_cost
            )

            kernel = np.ones((21, 21))
            result = (convolve2d(paths_corrected, kernel, mode='same') > 0).astype(np.float32)

            if cw == mosaic_height_index:
                ch += 1
                cw = 0

            result_image.paste(Image.fromarray(result), (cw * window_size, ch * window_size))

            cw += 1

        matrix = np.array(result_image.getdata())
        matrix = matrix.reshape((mosaic_width_index * window_size, mosaic_height_index * window_size))

        matrix = matrix[256:offset_array.shape[0], 256:offset_array.shape[1]]

        matrix = np.clip(matrix + array, 0, 1)

        plt.imshow(matrix)
        plt.show()

        with rio.open(img_path.replace(".tif", "") + "_roads.tif", "w+", **tif.profile) as dst:
            dst.write(np.expand_dims(matrix, axis=0))


if __name__ == '__main__':
    connect_animal_paths()
