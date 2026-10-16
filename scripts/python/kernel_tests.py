import numpy as np
import rasterio as rio
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt


def create_large_sobel_kernel(size):
    assert size % 2 == 1, "Kernel size must be odd"
    range_vals = np.linspace(-(size // 2), size // 2, size)
    sobel_x = np.outer(1 - np.abs(range_vals) / (size // 2), np.ones(size))
    sobel_y = sobel_x.T

    sobel_x = sobel_x / np.sum(np.abs(sobel_x))  # Normalize
    sobel_y = sobel_y / np.sum(np.abs(sobel_y))  # Normalize

    return sobel_x, sobel_y


def test():
    with rio.open("E:/master/stolsnek/part1/rgb.tif") as src:
        tif = torch.tensor(src.read(), dtype=torch.float32)[0:3, :, :]  # Read RGB channels
        profile = src.profile

        tif = tif.unsqueeze(0)

        sobel_x_np, sobel_y_np = create_large_sobel_kernel(7)

        sobel_x = torch.tensor(sobel_x_np, dtype=torch.float32)
        sobel_y = torch.tensor(sobel_y_np, dtype=torch.float32)

        output_x = F.conv2d(tif, sobel_x.repeat(3, 1, 1, 1), stride=1, padding=1, groups=3)
        output_y = F.conv2d(tif, sobel_y.repeat(3, 1, 1, 1), stride=1, padding=1, groups=3)

        sobel_output = torch.sigmoid(torch.sqrt(output_x ** 2 + output_y ** 2))
        sobel_output = sobel_output.squeeze(0)

        sobel_output_np = sobel_output.numpy()

        profile.update(dtype='float32', count=3)

        with rio.open("E:/master/stolsnek/part1/rgb_sobel.tif", "w+", **profile) as dst:
            dst.write(sobel_output_np.astype('float32'))




if __name__ == '__main__':
    test()
