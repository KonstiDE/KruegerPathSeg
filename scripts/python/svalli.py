import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

from tqdm import tqdm


def method(data_dir, sites):
    files = os.listdir(data_dir)

    for file in files:
        if os.path.basename(file).replace(".tif", "") in sites:
            image_rio = rio.open(os.path.join(data_dir, file))
            image = rio.open(os.path.join(data_dir, file)).read((1, 2, 3))

            image = image.transpose((1, 2, 0))
            image[image == np.nan] = 0
            image = np.round(image).astype(np.uint8)

            s = 512

            padded = np.zeros(((image.shape[0] // s + 1) * s, (image.shape[1] // s + 1) * s, image.shape[2]), np.uint8)
            dummy = np.zeros(((image.shape[0] // s + 1) * s, (image.shape[1] // s + 1) * s, 1), np.uint8)

            padded[0:image.shape[0], 0:image.shape[1], :] = image

            for x in range(padded.shape[0] // s):
                for y in range(padded.shape[1] // s):
                    window = padded[x * s:(x + 1) * s, y * s:(y + 1) * s, :]
                    window_dummy = np.zeros(window.shape, np.uint8)

                    gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    valid_contours = []

                    for c in contours:
                        area = cv2.contourArea(c)
                        if 4 < area < 24:
                            valid_contours.append(c)

                    cv2.drawContours(window_dummy, valid_contours, -1, (255, 255, 255), cv2.FILLED)

                    gray_back = cv2.cvtColor(window_dummy, cv2.COLOR_RGB2GRAY)

                    dummy[x * s:(x + 1) * s, y * s:(y + 1) * s, 0] = gray_back

            dummy[dummy != 255] = 0

            plt.imshow(dummy)
            plt.show()

            dummy = dummy[:image.shape[0], :image.shape[1], :]

            p = image_rio.profile
            p.update(
                dtype=rio.uint8,
                count=1
            )

            with rio.open(os.path.join(
                    data_dir,
                    os.path.basename(file).replace(".tif", "") + "_foots.tif"
            ), "w", **p) as dst:
                dst.write(dummy.squeeze(2), 1)
                dst.close()





if __name__ == '__main__':
    method("data", ["advendalen"])
