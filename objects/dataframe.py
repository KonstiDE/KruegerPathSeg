import os
import shutup
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

shutup.please()


class DataFrame:
    def __init__(self, index_w, index_h, data_stack, overlay, src_part, classification, auxiliary):
        self.index_w = index_w
        self.index_h = index_h
        self.src_part = src_part
        self.data_stack = data_stack
        self.overlay = overlay
        self.classification = classification
        self.auxiliary = auxiliary

    def show_rgb(self):
        plt.imshow(self.data_stack.transpose((1, 2, 0))[:, :, 0:3])
        plt.show()

    def show_overlay(self):
        cs = [(random.random(), random.random(), random.random()) for _ in range(len(np.unique(self.overlay)))]
        cs[0] = (0, 0, 0)

        plt.imshow(self.overlay.squeeze(0), cmap=ListedColormap(cs))
        plt.show()


    def show_degree(self):
        cs = [(random.random(), random.random(), random.random()) for _ in range(6)]
        cs[0] = (0, 0, 0)

        plt.imshow(self.auxiliary[0], cmap=ListedColormap(cs), vmax=6, vmin=0)
        plt.colorbar()
        plt.show()


    def show_distance(self):
        plt.imshow(self.auxiliary[1], cmap="viridis")
        plt.show()

    @staticmethod
    def to_frame(np_frame):

        src_part, index_w, index_h = os.path.basename(np_frame.fid.name).split("_")[1:4]
        data_stack = np.stack((np_frame["red"], np_frame["green"], np_frame["blue"]))
        overlay = np_frame["gt"]
        degree = np_frame["degree"]
        distance = np_frame["distance"]

        return DataFrame(
            index_w=index_w,
            index_h=index_h,
            src_part=src_part,
            data_stack=data_stack,
            overlay=overlay,
            classification=0,
            auxiliary=np.stack([degree, distance])
        )

