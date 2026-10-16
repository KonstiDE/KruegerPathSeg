import os

import matplotlib.pyplot as plt
import numpy as np

import geopandas

from PIL import Image
from pdf2image import convert_from_path

from cmap import Colormap

from preprocessing.tools import pad_image_rgb


def crop(basepath):
    Image.MAX_IMAGE_PIXELS = None

    cm = Colormap('seaborn:mako').to_mpl()

    variants = ["unet", "denseunet", "aspp", "attention", "stacked"]
    versions = ["buffer", "cluster"]

    buffers = []
    clusters = []

    ground_truth = None
    ground_cross = None

    alpha = None

    rgb = None

    for variant in variants:
        for version in versions:
            image_path = os.path.join(basepath, f"mosaic_{variant}_{version}_clipped.tif")

            if version == "buffer":
                buffers.append(np.asarray(Image.open(image_path)))
            else:
                clusters.append(np.asarray(Image.open(image_path)))


    for i in range(len(buffers)):
        if rgb is None:
            rgb = np.asarray(Image.open(os.path.join(basepath, "rgb_clipped.tif")))

        if ground_truth is None:
            ground_truth = geopandas.read_file(os.path.join(basepath, "completelines_clipped.gpkg"))
            try:
                ground_cross = geopandas.read_file(os.path.join(basepath, "completecross_clipped.gpkg"))
            except:
                pass

        if alpha is None:
            alpha = geopandas.read_file(os.path.join(basepath, "alpha.gpkg"))

        buffers[i] = np.where(buffers[i] <= 0, np.nan, buffers[i])
        clusters[i] = np.where(clusters[i] <= 0, np.nan, clusters[i])

    legend = convert_from_path("graphics/legend_columns_mako-crop.pdf")[0].convert("RGB")
    legend_array = np.asarray(legend)
    legend_array = pad_image_rgb(legend_array, color=(255, 255, 255), border_width=.15)

    fig, axs = plt.subplots(5, 3, figsize=(18, 25))

    axs[0, 0].imshow(rgb)

    axs[1, 0] = alpha.boundary.plot(ax=axs[1, 0], color='black', linewidth=1.5)
    axs[1, 0] = alpha.plot(ax=axs[1, 0], color='#242424', alpha=.8)
    axs[1, 0] = ground_truth.plot(ax=axs[1, 0], color='cyan', linewidth=1)
    if ground_cross is not None:
        axs[1, 0] = ground_cross.plot(ax=axs[1, 0], color='#FFC20A')

    axs[2, 0].imshow(legend_array)

    axs[0, 1].imshow(buffers[0], cmap=cm)
    axs[1, 1].imshow(buffers[1], cmap=cm)
    axs[2, 1].imshow(buffers[2], cmap=cm)
    axs[3, 1].imshow(buffers[3], cmap=cm)
    axs[4, 1].imshow(buffers[4], cmap=cm)

    axs[0, 2].imshow(clusters[0], cmap=cm)
    axs[1, 2].imshow(clusters[1], cmap=cm)
    axs[2, 2].imshow(clusters[2], cmap=cm)
    axs[3, 2].imshow(clusters[3], cmap=cm)
    axs[4, 2].imshow(clusters[4], cmap=cm)

    titles = ["V1", "V2", "V3", "V4", "V5"]

    for i in range(5):
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()
        axs[i, 2].set_axis_off()

        axs[i, 1].set_title(titles[i] + " - (Buffer)", fontsize=20, pad=10)
        axs[i, 2].set_title(titles[i] + " - (Cluster)", fontsize=20, pad=10)

    axs[0, 0].set_title("RGB - Composite", fontsize=20, pad=10)
    axs[1, 0].set_title("Ground Truth", fontsize=20, pad=10)

    axs[3, 0].remove()
    axs[4, 0].remove()

    plt.tight_layout()
    plt.savefig("mosaic_comparisons_appendix_Svalbard_mako.pdf", dpi=600)
    plt.show()


if __name__ == '__main__':
    crop("E:/master/svalbard/")
