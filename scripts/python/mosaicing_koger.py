import os
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image


def find_max_from_x_y(x_ys, index=0 | 1):
    return max([int(i.split("_")[index]) for i in x_ys])


def filter_filename(files, contains_this):
    worthy_filenames = []

    for file in files:
        if contains_this in file:
            worthy_filenames.append(file)

    return worthy_filenames


def mosaic(path):
    files = os.listdir(path)

    image_size = (2048, 2048)

    # riverside5_6_8_11_merge_transparent_mosaic_group1_9_1_blair_raster_scale_2.44140625
    prefix = "riverside5_6_8_11_merge_transparent_mosaic_group1_"
    suffix = "_blair_raster_scale_2.44140625.jpg"

    worthy = filter_filename(files, prefix)
    stripped = [s.replace(prefix, "").replace(suffix, "") for s in worthy]

    x_max = find_max_from_x_y(stripped, 0)
    y_max = find_max_from_x_y(stripped, 1)

    image = Image.new(mode="RGB", size=(image_size[0] * x_max, image_size[1] * y_max))

    for x in range(1, x_max + 1):
        for y in range(1, y_max + 1):
            try:
                tile = Image.open(os.path.join(path, prefix + str(x) + "_" + str(y) + suffix))

                image.paste(tile, (x * image_size[0], y * image_size[1]))

            except FileNotFoundError as _:
                pass

    plt.imshow(image, cmap="viridis")
    plt.colorbar()
    plt.show()

    image.save(prefix + suffix + "_mosaic.png")


if __name__ == '__main__':
    mosaic("E:/koger/data-repo/kenyan-ungulates/gametrail-annotations/data/all_data/images")
