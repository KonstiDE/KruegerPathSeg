import os

import scipy

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pdf2image import convert_from_path

from preprocessing.tools import pad_image_rgb


def crop(basepath, image_name):
    variants = ["unet", "denseunet", "aspp", "attention", "stacked"]
    versions = ["buffer", "cluster"]

    buffers = []
    clusters = []

    ground_truth_buffer = None
    ground_truth_cluster = None
    rgb = None

    for variant in variants:
        for version in versions:
            image_path = os.path.join(basepath, f"{variant}_{version}", image_name)

            if version == "buffer":
                buffers.append(np.asarray(Image.open(image_path).convert('RGB')))
            else:
                clusters.append(np.asarray(Image.open(image_path).convert('RGB')))


    for i in range(len(buffers)):
        if rgb is None:
            rgb = buffers[i][61:445, 239:623, :]
            rgb = scipy.ndimage.zoom(rgb, zoom=0.855, order=1)

        if ground_truth_buffer is None:
            ground_truth_buffer = buffers[i][89:417, 1211:1539, :]

        if ground_truth_cluster is None:
            ground_truth_cluster = clusters[i][89:417, 1211:1539, :]

        buffers[i] = buffers[i][89:417, 718:1046, :]
        clusters[i] = clusters[i][89:417, 718:1046, :]

    legend = convert_from_path("graphics/legend_rows-crop.pdf")[0].convert("RGB")
    legend_array = np.asarray(legend)
    legend_array = pad_image_rgb(legend_array, color=(255, 255, 255), border_width=.05)

    fig, axs = plt.subplots(2, 7, figsize=(34, 10))

    axs[0, 0].imshow(rgb)
    axs[1, 0].imshow(legend_array)

    axs[0, 1].imshow(ground_truth_buffer)
    axs[1, 1].imshow(ground_truth_cluster)

    axs[0, 2].imshow(buffers[0])
    axs[0, 3].imshow(buffers[1])
    axs[0, 4].imshow(buffers[2])
    axs[0, 5].imshow(buffers[3])
    axs[0, 6].imshow(buffers[4])

    axs[1, 2].imshow(clusters[0])
    axs[1, 3].imshow(clusters[1])
    axs[1, 4].imshow(clusters[2])
    axs[1, 5].imshow(clusters[3])
    axs[1, 6].imshow(clusters[4])

    titles = ["RGB composite", "Ground truth", "UNet - V1", "DenseUNet - V2", "ASPP - V3", "Attention - V4", "Stacked - V5"]

    for i in range(7):
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()

        axs[0, i].set_title(titles[i], fontsize=20)

    plt.rcParams['xtick.color'] = "white"
    plt.rcParams['ytick.color'] = "white"

    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig("single_tile_comparison3.pdf", dpi=600)
    plt.show()


if __name__ == '__main__':
    crop(
        "C:/Users/s371513/Desktop/master_results/",
        "part1_19_31_stolsnek.png"
    )
